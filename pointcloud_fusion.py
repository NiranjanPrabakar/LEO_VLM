#!/usr/bin/env python3
"""
pointcloud_fusion.py  —  RGB+Depth Fusion → Point Cloud + Viewpoint Vectors
=============================================================================

Integrates directly with your existing depth_mapper.py pipeline.
Same SPACE/F/ENTER capture controls. Reuses CapturedFrame objects and
YOLO detections from depth_mapper.InteractivePanoramicMapper.

Run:
    python pointcloud_fusion.py

Saves to results/:
    pointcloud_TIMESTAMP.ply              ← open in MeshLab / CloudCompare
    viewpoints_TIMESTAMP.json             ← per-frame camera pos + vectors
    bed_side_distances_TIMESTAMP.json     ← head/foot/left/right distances
    visualisation_TIMESTAMP/frame_XX.png  ← RGB with arrows overlaid

Coordinate system (CAMERA FRAME per-frame):
    X = right,  Y = down,  Z = forward (into scene)
    All distances in METRES.

Bed-side labels in JSON:
    head / foot / left / right  — bed-relative (uses semantic_depth_fusion
                                   orientation logic to remap from cam zones)
"""

from __future__ import annotations
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import struct
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Your existing modules ──────────────────────────────────────────
from depth_mapper import InteractivePanoramicMapper, CapturedFrame

# Import from semantic_depth_fusion — requires v10.
# If you see ImportError here, replace your semantic_depth_fusion.py
# with the v10 file from outputs/ first.
try:
    from semantic_depth_fusion import (
        detect_head_cam,
        build_cam_zones,
        build_remap,
        get_camera_view,
        MultiViewFusion,
        ZONE_PAD_END,
        ZONE_PAD_SIDE,
    )
except ImportError as e:
    print(f"\n[ERROR] {e}")
    print("  → Replace your semantic_depth_fusion.py with the v10 version")
    print("    (the one from the outputs/ folder Claude provided)")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

RESULTS_DIR  = Path("results")
MODEL_PATH   = Path("models/bed_detector_v1/weights/best.pt")

MAX_DEPTH_M  = 5.0      # clip beyond this
MIN_DEPTH_M  = 0.10     # ignore closer than this
STRIDE       = 3        # pixel stride for point cloud (3 = every 3rd pixel)
VOXEL_SIZE   = 0.015    # 1.5 cm voxel for merged cloud downsampling

# Zone sampling depth percentile
ZONE_PCT = 50.0


# ─────────────────────────────────────────────────────────
# RealSense deprojection helpers
# ─────────────────────────────────────────────────────────

class Intrinsics:
    """Thin wrapper around rs.intrinsics for vectorised deprojection."""

    def __init__(self, rs_intr):
        self.fx = float(rs_intr.fx)
        self.fy = float(rs_intr.fy)
        self.cx = float(rs_intr.ppx)
        self.cy = float(rs_intr.ppy)
        self.w  = int(rs_intr.width)
        self.h  = int(rs_intr.height)
        self._rs = rs_intr

    # ── Vectorised ────────────────────────────────────────
    def deproject(self,
                  u: np.ndarray,
                  v: np.ndarray,
                  d: np.ndarray) -> np.ndarray:
        """
        Deproject pixel arrays → 3-D camera-frame points.
        u, v, d: 1-D arrays of same length.  d in METRES.
        Returns (N, 3) float32.
        """
        X = (u - self.cx) * d / self.fx
        Y = (v - self.cy) * d / self.fy
        Z = d
        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    # ── Single pixel ──────────────────────────────────────
    def deproject_px(self, u: float, v: float, d: float) -> np.ndarray:
        pt = rs.rs2_deproject_pixel_to_point(self._rs, [float(u), float(v)], float(d))
        return np.array(pt, dtype=np.float32)

    def to_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy,
                    width=self.w, height=self.h)


# ─────────────────────────────────────────────────────────
# RGB + Depth → coloured point cloud (single frame)
# ─────────────────────────────────────────────────────────

def frame_to_pointcloud(
    rgb:   np.ndarray,          # (H,W,3) uint8 BGR
    depth: np.ndarray,          # (H,W)   uint16 mm
    intr:  Intrinsics,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        points  (N,3) float32  — XYZ in camera frame, metres
        colors  (N,3) uint8    — BGR
    """
    h, w = depth.shape

    # Pixel grid (subsampled)
    us = np.arange(0, w, stride, dtype=np.float32)
    vs = np.arange(0, h, stride, dtype=np.float32)
    ug, vg = np.meshgrid(us, vs)
    ug = ug.ravel();  vg = vg.ravel()
    ui = ug.astype(np.int32);  vi = vg.astype(np.int32)

    # Depth (mm → m), mask invalid
    d_m = depth[vi, ui].astype(np.float32) / 1000.0
    ok  = (d_m > MIN_DEPTH_M) & (d_m < MAX_DEPTH_M)
    ug, vg, d_m = ug[ok], vg[ok], d_m[ok]
    ui, vi = ui[ok], vi[ok]

    pts  = intr.deproject(ug, vg, d_m)
    cols = rgb[vi, ui]          # BGR colours at those pixels

    return pts, cols


# ─────────────────────────────────────────────────────────
# Voxel downsampling (no open3d needed)
# ─────────────────────────────────────────────────────────

def voxel_downsample(
    pts: np.ndarray,
    cols: np.ndarray,
    voxel: float = VOXEL_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return pts, cols
    idx = np.floor(pts / voxel).astype(np.int32)
    # Pack (ix, iy, iz) into a single int64 key
    # shift by large offsets so negatives work
    OFF = 10000
    keys = ((idx[:, 0] + OFF).astype(np.int64) * 100003 +
            (idx[:, 1] + OFF).astype(np.int64) * 1003 +
            (idx[:, 2] + OFF).astype(np.int64))
    unique_keys, inv = np.unique(keys, return_inverse=True)
    n = len(unique_keys)
    sum_pts  = np.zeros((n, 3), dtype=np.float64)
    sum_cols = np.zeros((n, 3), dtype=np.float64)
    cnt      = np.zeros(n, dtype=np.int64)
    np.add.at(sum_pts,  inv, pts)
    np.add.at(sum_cols, inv, cols.astype(np.float64))
    np.add.at(cnt,      inv, 1)
    out_pts  = (sum_pts  / cnt[:, None]).astype(np.float32)
    out_cols = (sum_cols / cnt[:, None]).astype(np.uint8)
    return out_pts, out_cols


# ─────────────────────────────────────────────────────────
# Binary PLY writer
# ─────────────────────────────────────────────────────────

def write_ply(pts: np.ndarray, cols: np.ndarray, path: Path):
    """Write binary-little-endian PLY. cols is BGR → written as RGB."""
    n = len(pts)
    rgb = cols[:, ::-1].astype(np.uint8)          # BGR → RGB
    hdr = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()
    # Build binary body as one numpy structured array
    dt = np.dtype([
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('r','u1'), ('g','u1'), ('b','u1'),
    ])
    body = np.empty(n, dtype=dt)
    body['x'] = pts[:, 0];  body['y'] = pts[:, 1];  body['z'] = pts[:, 2]
    body['r'] = rgb[:, 0];  body['g'] = rgb[:, 1];  body['b'] = rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(body.tobytes())
    print(f"  [ply]  Saved {path.name}  ({n:,} pts)")


# ─────────────────────────────────────────────────────────
# Per-frame viewpoint vector computation
# ─────────────────────────────────────────────────────────

def sample_zone_depth(
    depth: np.ndarray,
    zone:  Tuple[int, int, int, int],
    pct:   float = ZONE_PCT,
) -> Optional[float]:
    """
    Sample the median depth (in metres) in a bbox zone.
    Returns None if fewer than 50 valid pixels.
    """
    x1, y1, x2, y2 = zone
    h, w = depth.shape
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w,x2), min(h,y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth[y1:y2, x1:x2]
    valid  = region[(region > int(MIN_DEPTH_M*1000)) &
                    (region < int(MAX_DEPTH_M*1000))]
    if len(valid) < 50:
        return None
    return float(np.percentile(valid, pct)) / 1000.0   # mm → m


def zone_centre_px(zone: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = zone
    return ((x1+x2)/2.0, (y1+y2)/2.0)


def compute_viewpoint(
    frame:      CapturedFrame,
    bed_bbox:   List[float],
    intr:       Intrinsics,
    world_head_cam: str,
) -> dict:
    """
    For one CapturedFrame return a dict with:
      - camera_origin           : [0,0,0] always (camera frame)
      - bed_centre_3d           : XYZ of bed centre (metres, camera frame)
      - bed_side_distances_cam  : distance to each cam-frame zone edge
      - bed_side_vectors_cam    : unit vector from camera to each cam-frame edge
      - bed_side_distances_bed  : remapped to head/foot/left/right (metres)
      - bed_side_vectors_bed    : same but bed-relative names
      - camera_to_bed_m         : scalar distance camera → bed centre
      - viewing_angle_deg       : angle of camera direction vs forward axis
      - head_cam                : detected head direction in this frame
      - cam_view                : foot_end / left_side / right_side / head_end
      - remap                   : {cam_zone → bed_side}
    """
    depth = frame.depth
    h, w  = depth.shape

    # ── Detect orientation for THIS frame ──────────────────────────
    head_cam = detect_head_cam(bed_bbox, frame.detections) or world_head_cam
    cam_view = get_camera_view(bed_bbox, w, h, head_cam)
    remap    = build_remap(head_cam, cam_view)
    zones    = build_cam_zones(bed_bbox, w, h, head_cam)

    # ── Bed centre 3D ───────────────────────────────────────────────
    bx1,by1,bx2,by2 = bed_bbox
    # Central 50% of bbox
    mx1 = bx1 + 0.25*(bx2-bx1);  mx2 = bx1 + 0.75*(bx2-bx1)
    my1 = by1 + 0.25*(by2-by1);  my2 = by1 + 0.75*(by2-by1)
    bed_ctr_zone = (int(mx1),int(my1),int(mx2),int(my2))
    d_ctr = sample_zone_depth(depth, bed_ctr_zone, pct=60.0)

    bed_centre_3d = None
    cam_to_bed_m  = None
    view_angle    = None
    if d_ctr:
        cx_px = (bx1+bx2)/2.0;  cy_px = (by1+by2)/2.0
        bed_centre_3d = intr.deproject_px(cx_px, cy_px, d_ctr)
        cam_to_bed_m  = float(np.linalg.norm(bed_centre_3d))
        # Viewing angle vs camera forward axis (0,0,1)
        if cam_to_bed_m > 0:
            unit = bed_centre_3d / cam_to_bed_m
            cos_a = float(np.clip(unit[2], -1.0, 1.0))   # dot with (0,0,1)
            view_angle = round(float(np.degrees(np.arccos(cos_a))), 2)

    # ── Per-zone distances and vectors (camera frame) ───────────────
    cam_distances: Dict[str, Optional[float]]       = {}
    cam_vectors:   Dict[str, Optional[List[float]]] = {}

    for cz_name, zone in zones.items():
        d_m = sample_zone_depth(depth, zone)
        if d_m is None:
            cam_distances[cz_name] = None
            cam_vectors[cz_name]   = None
            continue
        u_c, v_c = zone_centre_px(zone)
        pt_3d = intr.deproject_px(u_c, v_c, d_m)
        dist  = float(np.linalg.norm(pt_3d))
        unit  = (pt_3d / dist).tolist() if dist > 0 else [0.0,0.0,1.0]
        cam_distances[cz_name] = round(float(dist), 3)
        cam_vectors[cz_name]   = [round(v,4) for v in unit]

    # ── Remap cam-frame → bed-frame ────────────────────────────────
    # remap = {cam_zone: bed_side}
    bed_distances: Dict[str, Optional[float]]       = {}
    bed_vectors:   Dict[str, Optional[List[float]]] = {}
    for cz, bs in remap.items():
        bed_distances[bs] = cam_distances.get(cz)
        bed_vectors[bs]   = cam_vectors.get(cz)

    return {
        'frame_index':           frame.index,
        'timestamp':             frame.timestamp,
        'head_cam':              head_cam,
        'cam_view':              cam_view,
        'remap':                 remap,
        'camera_origin':         [0.0, 0.0, 0.0],
        'bed_centre_3d':         bed_centre_3d.tolist() if bed_centre_3d is not None else None,
        'camera_to_bed_m':       round(cam_to_bed_m, 3) if cam_to_bed_m else None,
        'viewing_angle_deg':     view_angle,
        # Camera-frame zone results (raw)
        'cam_zone_distances_m':  cam_distances,
        'cam_zone_vectors_unit': cam_vectors,
        # Bed-relative results (what the robot uses)
        'bed_side_distances_m':  bed_distances,
        'bed_side_vectors_unit': bed_vectors,
    }


# ─────────────────────────────────────────────────────────
# Aggregate distances across frames
# ─────────────────────────────────────────────────────────

def aggregate_distances(viewpoints: List[dict]) -> Dict[str, dict]:
    """
    Aggregate bed_side_distances_m across all frames.
    Returns {side: {median_m, min_m, max_m, std_m, n, all_m}}
    """
    buckets: Dict[str, List[float]] = {
        s:[] for s in ['head','foot','left','right']
    }
    for vp in viewpoints:
        for side, d in (vp.get('bed_side_distances_m') or {}).items():
            if d is not None and d > 0:
                buckets[side].append(d)

    out = {}
    for side, vals in buckets.items():
        if vals:
            out[side] = {
                'median_m':   round(float(np.median(vals)), 3),
                'min_m':      round(float(np.min(vals)),    3),
                'max_m':      round(float(np.max(vals)),    3),
                'std_m':      round(float(np.std(vals)),    3),
                'n_frames':   len(vals),
                'all_m':      [round(v,3) for v in vals],
            }
        else:
            out[side] = {
                'median_m': None, 'min_m': None,
                'max_m': None,    'std_m': None,
                'n_frames': 0,    'all_m': [],
            }
    return out


# ─────────────────────────────────────────────────────────
# Visualisation overlay
# ─────────────────────────────────────────────────────────

ARROW_COLORS = {
    'cam_top':    (255, 100,  50),   # blue
    'cam_bottom': ( 50, 255,  80),   # green
    'cam_left':   ( 50,  80, 255),   # red
    'cam_right':  (255, 255,   0),   # cyan
}
BED_SIDE_COLORS = {
    'head':  (255, 80,  80),
    'foot':  (80, 255,  80),
    'left':  (80,  80, 255),
    'right': (255, 255,  0),
}

def draw_overlay(
    rgb:       np.ndarray,
    bed_bbox:  List[float],
    vp:        dict,
) -> np.ndarray:
    """Draw bed bbox, distance arrows, and info panel on RGB frame."""
    vis = rgb.copy()
    h, w = vis.shape[:2]

    # Bed bbox
    x1,y1,x2,y2 = map(int, bed_bbox)
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), 2)
    cx,cy = (x1+x2)//2, (y1+y2)//2
    cv2.drawMarker(vis, (cx,cy), (0,255,255), cv2.MARKER_CROSS, 16, 2)

    # Edge anchor points
    edge_anchors = {
        'cam_top':    (cx, y1),
        'cam_bottom': (cx, y2),
        'cam_left':   (x1, cy),
        'cam_right':  (x2, cy),
    }
    # Remap for label colour
    remap = vp.get('remap', {})   # {cam_zone: bed_side}

    cam_dists = vp.get('cam_zone_distances_m', {})

    for cz, anchor in edge_anchors.items():
        d    = cam_dists.get(cz)
        col  = ARROW_COLORS.get(cz, (200,200,200))
        bed_side = remap.get(cz, cz)
        bed_col  = BED_SIDE_COLORS.get(bed_side, col)

        # Arrow from centre to edge
        cv2.arrowedLine(vis, (cx,cy), anchor, bed_col, 2, tipLength=0.12)

        if d is not None:
            # Label at edge
            lx = int(anchor[0]); ly = int(anchor[1])
            lx = max(5, min(w-130, lx - 55))
            ly = max(18, min(h-5,  ly + (20 if 'bottom' in cz else -8)))
            label = f"{bed_side}: {d:.2f}m"
            cv2.rectangle(vis, (lx-2, ly-14), (lx+128, ly+3), (0,0,0), -1)
            cv2.putText(vis, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bed_col,
                        1, cv2.LINE_AA)

    # Info panel (top-left)
    lines = [
        f"Frame {vp['frame_index']}  |  {vp['timestamp']}",
        f"head_cam={vp['head_cam']}  cam_view={vp['cam_view']}",
        f"cam→bed: {vp['camera_to_bed_m']}m  "
        f"angle: {vp['viewing_angle_deg']}°",
    ]
    for i, ln in enumerate(lines):
        y = 18 + i*18
        cv2.rectangle(vis, (0, y-13), (w, y+4), (0,0,0), -1)
        cv2.putText(vis, ln, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200),
                    1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────

def run():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Phase 1: Panoramic capture (reuses your existing mapper) ──────
    print("\n" + "▓"*50)
    print("▓  POINT CLOUD FUSION — Panoramic Capture")
    print("▓"*50)

    mapper = InteractivePanoramicMapper(model_path=str(MODEL_PATH))

    # Grab intrinsics before run() closes the window
    profile = mapper.pipeline.get_active_profile()
    col_str = profile.get_stream(rs.stream.color)
    intr    = Intrinsics(
        col_str.as_video_stream_profile().get_intrinsics()
    )
    print(f"\n  Camera intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.cx:.1f} cy={intr.cy:.1f}")

    # Run capture — returns accessibility, stats, objects, frames
    accessibility, stats, objects, frames = mapper.run()

    print(f"\n  Captured {len(frames)} frames")
    print(f"  Accessibility: {accessibility}")

    # ── Phase 2: Find best bed bbox ───────────────────────────────────
    best_bbox = None
    best_conf = 0.0
    for f in frames:
        bed = f.detections.get('bed')
        if bed and bed.get('confidence', 0) > best_conf:
            best_conf = bed['confidence']
            best_bbox = bed['bbox']

    if best_bbox is None:
        print("  [WARN] No bed detected — using frame centre as fallback")
        h, w = frames[0].depth.shape
        best_bbox = [w*0.10, h*0.05, w*0.90, h*0.95]

    print(f"  Best bed bbox: {[round(v,1) for v in best_bbox]}  "
          f"conf={best_conf:.2f}")

    # ── Phase 3: Establish world head direction ───────────────────────
    orient_votes = []
    for f in frames:
        hc = detect_head_cam(best_bbox, f.detections)
        if hc:
            orient_votes.append(hc)

    from collections import Counter
    if orient_votes:
        world_head_cam = Counter(orient_votes).most_common(1)[0][0]
    else:
        world_head_cam = 'top'
    print(f"  World head direction: '{world_head_cam}'  "
          f"(votes: {dict(Counter(orient_votes))})")

    # ── Phase 4: Per-frame point clouds + viewpoint vectors ───────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = RESULTS_DIR / f"visualisation_{ts}"
    vis_dir.mkdir(exist_ok=True)

    all_pts:  List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    viewpoints: List[dict]     = []

    print(f"\n  Processing {len(frames)} frames…")
    for f in frames:
        print(f"\n  ── Frame {f.index + 1}/{len(frames)}  "
              f"({f.timestamp}) ──")

        # Point cloud
        pts, cols = frame_to_pointcloud(f.rgb, f.depth, intr, stride=STRIDE)
        all_pts.append(pts);  all_cols.append(cols)
        print(f"     Points: {len(pts):,}")

        # Viewpoint vectors
        vp = compute_viewpoint(f, best_bbox, intr, world_head_cam)
        viewpoints.append(vp)

        # Print distances
        print(f"     cam→bed: {vp['camera_to_bed_m']}m  "
              f"angle: {vp['viewing_angle_deg']}°  "
              f"head_cam: {vp['head_cam']}  "
              f"cam_view: {vp['cam_view']}")
        print(f"     Remap: {vp['remap']}")
        print(f"     Bed-side distances:")
        for side in ['head','foot','left','right']:
            d = vp['bed_side_distances_m'].get(side)
            v = vp['bed_side_vectors_unit'].get(side)
            d_s = f"{d:.3f}m" if d else "  N/A "
            v_s = (f"[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"
                   if v else "    N/A    ")
            print(f"       {side:>5s}: {d_s}  unit_vec={v_s}")

        # Visualisation image
        vis_img = draw_overlay(f.rgb, best_bbox, vp)
        clipped   = np.clip(f.depth, 0, 4000).astype(np.float32)
        depth_col = cv2.applyColorMap(
            (clipped / 4000 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_col[f.depth == 0] = 0
        combined = np.hstack([vis_img, depth_col])
        cv2.imwrite(str(vis_dir / f"frame_{f.index:02d}.png"), combined)

    # ── Phase 5: Merge + downsample point cloud ───────────────────────
    print(f"\n  Merging point clouds…")
    merged_pts  = np.concatenate(all_pts,  axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)
    print(f"  Raw total: {len(merged_pts):,} points")

    merged_pts, merged_cols = voxel_downsample(
        merged_pts, merged_cols, voxel=VOXEL_SIZE
    )
    print(f"  After voxel downsample ({VOXEL_SIZE}m): {len(merged_pts):,} points")

    # ── Phase 6: Aggregate distances ─────────────────────────────────
    agg = aggregate_distances(viewpoints)

    # ── Phase 7: Save all results ─────────────────────────────────────
    print(f"\n  Saving to {RESULTS_DIR}/")

    # 1. Point cloud PLY
    ply_path = RESULTS_DIR / f"pointcloud_{ts}.ply"
    write_ply(merged_pts, merged_cols, ply_path)

    # 2. Per-frame viewpoints JSON
    vp_path = RESULTS_DIR / f"viewpoints_{ts}.json"
    with open(vp_path, 'w') as fp:
        json.dump({
            'timestamp':          ts,
            'num_frames':         len(frames),
            'world_head_cam':     world_head_cam,
            'best_bed_bbox':      best_bbox,
            'accessibility_map':  accessibility,
            'camera_intrinsics':  intr.to_dict(),
            'viewpoints':         viewpoints,
        }, fp, indent=2, default=_json_default)
    print(f"  [json] Saved {vp_path.name}")

    # 3. Bed-side distances summary JSON
    bd_path = RESULTS_DIR / f"bed_side_distances_{ts}.json"
    with open(bd_path, 'w') as fp:
        json.dump({
            'timestamp':     ts,
            'num_frames':    len(frames),
            'world_head_cam': world_head_cam,
            'best_bed_bbox': best_bbox,
            'accessibility': accessibility,
            'bed_side_distances': agg,
            'legend': {
                'head':  'headboard end of bed',
                'foot':  'foot end of bed (usually open)',
                'left':  'left long side (robot standing at foot, facing head)',
                'right': 'right long side',
                'distances': 'camera to that bed edge, in metres, per frame',
            },
        }, fp, indent=2, default=_json_default)
    print(f"  [json] Saved {bd_path.name}")

    # 4. Visualisation images already saved above
    print(f"  [vis]  {len(frames)} images → {vis_dir.name}/")

    # ── Phase 8: Print final summary ─────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  Frames           : {len(frames)}")
    print(f"  Point cloud pts  : {len(merged_pts):,}")
    print(f"  World head dir   : {world_head_cam}")
    print()
    print("  BED SIDE DISTANCES  (median across all frames):")
    icons = {'free':'✓','partially_blocked':'⚠','blocked':'✗','unknown':'?'}
    for side in ['head','foot','left','right']:
        a  = accessibility.get(side,'unknown')
        d  = agg[side]
        dm = f"{d['median_m']:.3f}m" if d['median_m'] else "  N/A "
        mn = f"{d['min_m']:.3f}" if d['min_m'] else "N/A"
        mx = f"{d['max_m']:.3f}" if d['max_m'] else "N/A"
        print(f"    {side.upper():>5s}:  {dm}  "
              f"(min={mn} max={mx} n={d['n_frames']})  "
              f"{icons[a]} {a}")
    print()
    print(f"  Files saved:")
    print(f"    {ply_path}")
    print(f"    {vp_path}")
    print(f"    {bd_path}")
    print(f"    {vis_dir}/  ({len(frames)} images)")
    print("="*60)

    mapper.cleanup()
    return {
        'ply':        ply_path,
        'viewpoints': vp_path,
        'distances':  bd_path,
        'vis_dir':    vis_dir,
    }


def _json_default(obj):
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32,   np.int64)):   return int(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
