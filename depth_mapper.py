#!/usr/bin/env python3
"""
depth_mapper.py  —  Interactive Panoramic Capture + Multi-View Fusion
======================================================================
v2 — SAM Integration
  SAM is now used to refine YOLO bounding boxes into pixel-precise masks.
  Every detection dict gains a 'sam_mask' key (np.ndarray H×W bool | None).
  All downstream code (semantic_depth_fusion) uses those masks when present
  and falls back silently to the original YOLO-box logic when SAM fails.

  SAM is loaded once at startup.  Image embedding is computed once per
  captured frame (batch refinement), so the per-detection overhead is tiny.

  Nothing else changed — NMS, reclassification, fusion, UI all identical.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple

# ── SAM integration ───────────────────────────────────────────────────
from sam_integration import SAMRefiner, draw_sam_masks


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

WALL_OFFSET_MM  = 600
FREE_FAR_PCT    = 0.30
BLOCKED_FAR_PCT = 0.55

CONF_THRESHOLD = {
    'Bed':                0.40,
    'Bed_Cover':          0.35,
    'Duvet':              0.35,
    'Duvet_Cover':        0.35,
    'Mattress_protector': 0.35,
    'Pillow':             0.40,
    'Pillow_cover':       0.35,
    'obstacles':          0.50,
    'Wall':               0.55,
    'unknown':            0.50,
}

NMS_IOU_THRESHOLD = 0.45

CLASS_MAP = {
    0: 'Bed', 1: 'Bed_Cover', 2: 'Duvet', 3: 'Duvet_Cover',
    4: 'Mattress_protector', 5: 'Pillow', 6: 'Pillow_cover',
    7: 'Wall', 8: 'obstacles',
}

SIDE_COLOR = {
    'free':             (  0, 220,   0),
    'partially_blocked':(  0, 180, 255),
    'blocked':          (  0,   0, 220),
    'unknown':          (160, 160, 160),
}

YOLO_COLORS = {
    'Bed':                (255, 200,   0),
    'Bed_Cover':          (  0, 200, 255),
    'Duvet':              (  0, 180, 255),
    'Duvet_Cover':        (  0, 160, 255),
    'Mattress_protector': (  0, 140, 255),
    'Pillow':             (180,   0, 255),
    'Pillow_cover':       (200,   0, 220),
    'obstacles':          (  0,   0, 255),
    'Wall':               (120, 120, 120),
    'unknown':            (200, 200, 200),
}


# ─────────────────────────────────────────────────────────
# NMS + reclassification helpers  (UNCHANGED from v1)
# ─────────────────────────────────────────────────────────

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def nms_per_class(objects: List[dict], iou_thresh: float = NMS_IOU_THRESHOLD) -> List[dict]:
    if not objects:
        return []
    by_class: Dict[str, List[dict]] = {}
    for obj in objects:
        by_class.setdefault(obj['class'], []).append(obj)
    kept = []
    for cls, boxes in by_class.items():
        boxes = sorted(boxes, key=lambda x: -x['confidence'])
        suppressed = [False] * len(boxes)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            for j in range(i + 1, len(boxes)):
                if suppressed[j]:
                    continue
                if compute_iou(boxes[i]['bbox'], boxes[j]['bbox']) > iou_thresh:
                    suppressed[j] = True
    return kept


def reclassify_walls_vs_obstacles(
    objects: List[dict],
    frame_w: int,
    frame_h: int,
) -> List[dict]:
    result = []
    for obj in objects:
        cls  = obj['class']
        x1, y1, x2, y2 = obj['bbox']
        w = x2 - x1
        h = y2 - y1
        if cls == 'Wall':
            spans_wide = (w / frame_w) > 0.35
            spans_tall = (h / frame_h) > 0.35
            near_edge  = (x1 < 20 or y1 < 20 or
                          x2 > frame_w - 20 or y2 > frame_h - 20)
            if not ((spans_wide or spans_tall) and near_edge):
                obj = dict(obj)
                obj['class']         = 'obstacles'
                obj['reclassified']  = True
        result.append(obj)
    return result


def filter_and_clean_detections(
    raw_boxes: list,
    frame_w:   int,
    frame_h:   int,
) -> dict:
    """
    Confidence filter → reclassify → NMS → split into categories.
    SAM masks are NOT added here; call sam_refiner.refine_batch() afterwards.
    """
    filtered = []
    for box in raw_boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls      = CLASS_MAP.get(cls_id, 'unknown')
        min_conf = CONF_THRESHOLD.get(cls, 0.40)
        if conf >= min_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            filtered.append({
                'bbox':       [x1, y1, x2, y2],
                'confidence': conf,
                'class':      cls,
                'sam_mask':   None,   # placeholder; filled by SAMRefiner.refine_batch()
            })

    filtered = reclassify_walls_vs_obstacles(filtered, frame_w, frame_h)
    filtered = nms_per_class(filtered)

    detections = {
        'bed': None, 'bedding': [], 'pillows': [],
        'obstacles': [], 'walls': []
    }
    for obj in filtered:
        cls = obj['class']
        if cls == 'Bed':
            if not detections['bed'] or obj['confidence'] > detections['bed']['confidence']:
                detections['bed'] = obj
        elif cls in ('Bed_Cover', 'Duvet', 'Duvet_Cover', 'Mattress_protector'):
            detections['bedding'].append(obj)
        elif cls in ('Pillow', 'Pillow_cover'):
            detections['pillows'].append(obj)
        elif cls == 'Wall':
            detections['walls'].append(obj)
        elif cls == 'obstacles':
            detections['obstacles'].append(obj)

    return detections


# ─────────────────────────────────────────────────────────
# Per-frame storage  (UNCHANGED)
# ─────────────────────────────────────────────────────────

class CapturedFrame:
    def __init__(self, rgb: np.ndarray, depth: np.ndarray,
                 detections: dict, index: int):
        self.rgb        = rgb
        self.depth      = depth
        self.detections = detections
        self.index      = index
        self.timestamp  = datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ─────────────────────────────────────────────────────────
# Multi-view depth fusion  (UNCHANGED — kept as fallback)
# ─────────────────────────────────────────────────────────

class MultiViewFusion:
    def fuse(
        self,
        frames:   List[CapturedFrame],
        bed_bbox: List[float],
    ) -> Tuple[Dict[str, str], Dict[str, dict]]:
        if not frames:
            return {s: 'unknown' for s in ['head','foot','left','right']}, {}

        side_depths:   Dict[str, List[float]] = {s: [] for s in ['head','foot','left','right']}
        side_far_pcts: Dict[str, List[float]] = {s: [] for s in ['head','foot','left','right']}

        for frame in frames:
            self._extract(frame.depth, bed_bbox, side_depths, side_far_pcts)

        accessibility: Dict[str, str]  = {}
        stats:         Dict[str, dict] = {}

        for side in ['head','foot','left','right']:
            depths   = side_depths[side]
            far_pcts = side_far_pcts[side]
            if not depths:
                accessibility[side] = 'unknown'
                stats[side] = {}
                continue
            med_depth   = float(np.median(depths))
            med_far_pct = float(np.median(far_pcts))
            confidence  = min(1.0, len(depths) / max(len(frames), 1))
            label       = self._classify(side, med_far_pct)
            accessibility[side] = label
            stats[side] = {
                'median_depth':   round(med_depth, 1),
                'median_depth_m': round(med_depth / 1000.0, 2),
                'far_pct':        round(med_far_pct, 3),
                'confidence':     round(confidence, 2),
                'num_views':      len(depths),
            }
        return accessibility, stats

    def _extract(self, depth, bed_bbox, side_depths, side_far_pcts):
        x1,y1,x2,y2 = map(int, bed_bbox)
        h, w = depth.shape
        x1,x2 = max(0,x1), min(w-1,x2)
        y1,y2 = max(0,y1), min(h-1,y2)
        bed_region = depth[y1:y2, x1:x2]
        valid_bed  = bed_region[bed_region > 0]
        if valid_bed.size < 100:
            return
        bed_depth = float(np.median(valid_bed))
        pad_x = int(0.30*(x2-x1)); pad_y = int(0.30*(y2-y1))
        regions = {
            'left':  depth[y1:y2,            max(0, x1-pad_x):x1],
            'right': depth[y1:y2,            x2:min(w, x2+pad_x)],
            'head':  depth[max(0, y1-pad_y):y1, x1:x2],
            'foot':  depth[y2:min(h, y2+pad_y), x1:x2],
        }
        for side, region in regions.items():
            valid = region[region > 0]
            if valid.size < 30:
                continue
            med   = float(np.median(valid))
            far_p = float(np.sum(valid > (bed_depth + WALL_OFFSET_MM)) / len(valid))
            side_depths[side].append(med)
            side_far_pcts[side].append(far_p)

    def _classify(self, side: str, far_pct: float) -> str:
        if side == 'head':
            if far_pct < 0.25:  return 'blocked'
            if far_pct < 0.55:  return 'partially_blocked'
            return 'free'
        elif side == 'foot':
            if far_pct < FREE_FAR_PCT:    return 'free'
            if far_pct < BLOCKED_FAR_PCT: return 'partially_blocked'
            return 'blocked'
        else:
            if far_pct < 0.20:  return 'blocked'
            if far_pct < 0.50:  return 'partially_blocked'
            return 'free'


# ─────────────────────────────────────────────────────────
# Object aggregation  (UNCHANGED)
# ─────────────────────────────────────────────────────────

def aggregate_objects(frames: List[CapturedFrame]) -> dict:
    seen      = set()
    bedding   = []
    pillows   = []
    obstacles = []
    walls     = []
    best_bed  = None
    best_conf = 0.0

    for frame in frames:
        det = frame.detections
        bed = det.get('bed')
        if bed and bed['confidence'] > best_conf:
            best_bed  = bed
            best_conf = bed['confidence']
        for o in det.get('bedding', []):
            if o['class'] not in seen:
                seen.add(o['class']); bedding.append(o)
        for o in det.get('pillows', []):
            if o['class'] not in seen:
                seen.add(o['class']); pillows.append(o)
        for o in det.get('obstacles', []):
            obstacles.append(o)
        for o in det.get('walls', []):
            if 'Wall' not in seen:
                seen.add('Wall'); walls.append(o)

    return {
        'bed': best_bed, 'bedding': bedding,
        'pillows': pillows, 'obstacles': obstacles, 'walls': walls,
    }


# ─────────────────────────────────────────────────────────
# Drawing helpers  (UPDATED — SAM masks overlaid when present)
# ─────────────────────────────────────────────────────────

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth, 0, 4000).astype(np.float32)
    norm    = (clipped / 4000.0 * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[depth == 0] = (0, 0, 0)
    return colored


def draw_yolo_boxes(rgb: np.ndarray, detections: dict) -> np.ndarray:
    """
    Draw SAM masks (semi-transparent fill) then YOLO boxes on top.
    A small ⬛ marker in the label shows when a SAM mask is active.
    """
    # 1. SAM mask overlay
    vis = draw_sam_masks(rgb, detections, alpha=0.30)

    # 2. YOLO bounding boxes
    all_objs = []
    for key in ('bed', 'bedding', 'pillows', 'obstacles', 'walls'):
        val = detections.get(key)
        if not val:
            continue
        if isinstance(val, list):
            all_objs.extend(val)
        else:
            all_objs.append(val)

    for obj in all_objs:
        x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
        cls     = obj.get('class', 'unknown')
        conf    = obj.get('confidence', 0.0)
        relab   = " *" if obj.get('reclassified') else ""
        has_sam = " S" if obj.get('sam_mask') is not None else ""
        color   = YOLO_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cls}{relab}{has_sam} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(vis, (x1, y1-th-bl-3), (x1+tw+3, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-bl-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def build_capture_hud(
    rgb_yolo:     np.ndarray,
    depth_color:  np.ndarray,
    num_captured: int,
    last_flash:   float,
    sam_panel:    np.ndarray = None,
) -> np.ndarray:
    h, w = rgb_yolo.shape[:2]
    flash = max(0.0, 1.0 - (time.time() - last_flash) * 5)
    if flash > 0:
        overlay = rgb_yolo.copy()
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (255, 255, 255), 14)
        cv2.addWeighted(overlay, flash, rgb_yolo, 1-flash, 0, rgb_yolo)

    if sam_panel is not None:
        # 3-panel layout: YOLO | SAM | Depth  — each scaled to w//3 * 3 total
        tw = w   # each panel same width as camera frame
        pw = tw * 3
        # resize all to same height
        sam_r   = cv2.resize(sam_panel,   (tw, h))
        depth_r = cv2.resize(depth_color, (tw, h))
        row_cams = np.hstack([rgb_yolo, sam_r, depth_r])
        banner = np.full((55, pw, 3), 20, dtype=np.uint8)
        cv2.putText(banner, "LEFT: YOLO boxes     CENTRE: SAM masks     RIGHT: Depth",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(banner,
                    f"SPACE=capture ({num_captured} taken)   u=undo   f/ENTER=finish & analyse   q=quit",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        dots = np.full((26, pw, 3), 12, dtype=np.uint8)
    else:
        row_cams = np.hstack([rgb_yolo, depth_color])
        banner = np.full((55, w*2, 3), 20, dtype=np.uint8)
        cv2.putText(banner, "PANORAMIC CAPTURE  —  Move camera around the bed",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(banner,
                    f"SPACE=capture ({num_captured} taken)   u=undo   f/ENTER=finish & analyse   q=quit",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        dots = np.full((26, w*2, 3), 12, dtype=np.uint8)

    total_w = row_cams.shape[1]
    for i in range(num_captured):
        cx = 14 + i * 24
        if cx < total_w - 12:
            cv2.circle(dots, (cx, 13), 9, (0, 200, 80), -1)
            cv2.putText(dots, str(i+1), (cx-5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([banner, row_cams, dots])


def build_result_panel(
    accessibility: dict,
    stats:         dict,
    objects:       dict,
    num_frames:    int,
) -> np.ndarray:
    W, H = 1280, 310
    panel = np.full((H, W, 3), 18, dtype=np.uint8)
    cv2.putText(panel, f"SCAN COMPLETE  —  {num_frames} frames  [YOLO+SAM fusion]",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)
    labels_map = {'free':'FREE','partially_blocked':'PARTIAL',
                  'blocked':'BLOCKED','unknown':'UNKNOWN'}
    cw = W // 4
    for i, side in enumerate(['head','foot','left','right']):
        status = accessibility.get(side, 'unknown')
        color  = SIDE_COLOR[status]
        s      = stats.get(side, {})
        d_m    = s.get('median_depth_m', 0.0)
        conf   = s.get('confidence', 0.0)
        nv     = s.get('num_views', 0)
        x      = i * cw
        bg = panel.copy()
        cv2.rectangle(bg, (x+4, 48), (x+cw-4, 185), color, -1)
        cv2.addWeighted(bg, 0.18, panel, 0.82, 0, panel)
        cv2.rectangle(panel, (x+4, 48), (x+cw-4, 185), color, 2)
        cx  = x + cw // 2
        lbl = labels_map[status]
        cv2.putText(panel, side.upper(), (cx-28, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(panel, lbl, (cx-len(lbl)*7, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"{d_m:.2f}m", (cx-22, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(panel, f"conf={conf:.2f}  {nv} views", (cx-42, 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150,150,150), 1, cv2.LINE_AA)
    obj_names = []
    if objects.get('bed'):               obj_names.append('Bed')
    for o in objects.get('bedding', []): obj_names.append(o['class'])
    for o in objects.get('pillows', []): obj_names.append(o['class'])
    for o in objects.get('obstacles',[]): obj_names.append('obstacle')
    cv2.putText(panel, "Objects: " + (", ".join(obj_names) if obj_names else "none"),
                (10, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,180), 1, cv2.LINE_AA)
    best = None
    for side in ['foot','left','right','head']:
        if accessibility.get(side) == 'free':
            best = side; break
    if not best:
        for side in ['foot','left','right','head']:
            if accessibility.get(side) == 'partially_blocked':
                best = side; break
    if best:
        cv2.putText(panel, f"Best approach: {best.upper()}",
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Starting task planning automatically…",
                (10, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140,140,140), 1, cv2.LINE_AA)
    return panel


# ─────────────────────────────────────────────────────────
# Main mapper class  (UPDATED — SAM added)
# ─────────────────────────────────────────────────────────

class InteractivePanoramicMapper:
    def __init__(
        self,
        model_path:     str = "models/bed_detector_v1/weights/best.pt",
        sam_model_type: str = "vit_b",   # "vit_b" | "vit_h"
        sam_device:     str = "cpu",
        sam:            object = None,   # pass already-loaded SAMRefiner to skip reload
    ):
        print("\n" + "="*60)
        print("  INTERACTIVE PANORAMIC DEPTH MAPPER  [YOLO + SAM]")
        print("="*60)

        print("\n[1/3] Starting RealSense camera…")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        print("      ✓ Camera streaming")

        print("\n[2/3] Loading YOLO…")
        self.yolo = YOLO(model_path)
        print("      ✓ YOLO ready")

        print("\n[3/3] SAM…")
        if sam is not None:
            # Reuse already-loaded instance — no second load
            self.sam     = sam
            self._sam_ok = True
            print("      ✓ SAM reused (already loaded)")
        else:
            try:
                self.sam     = SAMRefiner(model_type=sam_model_type, device=sam_device)
                self._sam_ok = True
            except Exception as e:
                print(f"      ⚠ SAM failed to load ({e})")
                print("      ↳ Continuing with YOLO-only (bounding boxes only)")
                self.sam     = None
                self._sam_ok = False

        self.fusion = MultiViewFusion()
        self.frames: List[CapturedFrame] = []
        self._last_flash = -10.0
        self._live_detections = {
            'bed':None,'bedding':[],'pillows':[],'obstacles':[],'walls':[]
        }
        self._frame_counter = 0
        # Window pre-created in main() before loading; just pump the event loop
        cv2.waitKey(1)

    # ── SAM refinement helper ─────────────────────────────

    def _apply_sam(self, rgb: np.ndarray, detections: dict) -> dict:
        """Add SAM masks to all detections.  No-op if SAM unavailable."""
        if not self._sam_ok or self.sam is None:
            return detections
        try:
            return self.sam.refine_batch(rgb, detections)
        except Exception as e:
            print(f"[SAM] refine_batch error: {e} — skipping")
            return detections

    def _get_frame(self):
        try:
            frames  = self.pipeline.wait_for_frames(timeout_ms=200)
            aligned = self.align.process(frames)
            color   = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color or not depth_f:
                return None, None
            return (np.asanyarray(color.get_data()),
                    np.asanyarray(depth_f.get_data()))
        except Exception:
            return None, None

    def _run_yolo(self, rgb: np.ndarray) -> dict:
        results = self.yolo(rgb, verbose=False)[0]
        h, w = rgb.shape[:2]
        return filter_and_clean_detections(results.boxes, w, h)

    def _best_bed_bbox(self):
        best_conf = 0.0; best_bbox = None
        for frame in self.frames:
            bed = frame.detections.get('bed')
            if bed and bed['confidence'] > best_conf:
                best_conf = bed['confidence']
                best_bbox = bed['bbox']
        return best_bbox

    def run(self) -> Tuple[Dict, Dict, Dict, List[CapturedFrame]]:
        print("\n" + "─"*60)
        print("INSTRUCTIONS:")
        print("  Move camera slowly around the bed:")
        print("  → foot end  → left side  → right side  → slight from above")
        print("  Press SPACE to snapshot each position (aim for 5–9 snaps)")
        print("  Press F or ENTER when done — analysis runs automatically")
        print("─"*60 + "\n")

        self._last_sam_panel = None   # updated by background SAM thread
        self._sam_thread_running = False

        # ── Background SAM thread ─────────────────────────
        import threading
        _sam_lock   = threading.Lock()
        _stop_sam   = threading.Event()
        _latest_rgb_for_sam = [None]
        _latest_det_for_sam = [None]

        def _sam_worker():
            while not _stop_sam.is_set():
                with _sam_lock:
                    rgb_s = _latest_rgb_for_sam[0]
                    det_s = _latest_det_for_sam[0]
                if rgb_s is None or det_s is None:
                    import time; time.sleep(0.05); continue
                try:
                    refined = self._apply_sam(rgb_s, det_s)
                    panel   = draw_sam_masks(rgb_s, refined, alpha=0.55)
                    # Burn class labels onto the SAM panel
                    for grp in ('bed','bedding','pillows','obstacles','walls'):
                        val = refined.get(grp)
                        items = [val] if isinstance(val,dict) and val else (val or [])
                        for obj in items:
                            if obj.get('sam_mask') is None: continue
                            x1,y1 = int(obj['bbox'][0]), int(obj['bbox'][1])
                            cv2.putText(panel, obj.get('class','?'),
                                (x1+2, y1+14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255,255,255), 1, cv2.LINE_AA)
                    self._last_sam_panel = panel
                except Exception as e:
                    pass
                import time; time.sleep(0.05)   # don't spin

        sam_bg = threading.Thread(target=_sam_worker, daemon=True)
        sam_bg.start()
        # ─────────────────────────────────────────────────

        while True:
            rgb, depth = self._get_frame()
            if rgb is None:
                continue
            self._frame_counter += 1

            # Live YOLO every 4th frame
            if self._frame_counter % 4 == 0:
                self._live_detections = self._run_yolo(rgb)
                # Feed latest frame+detections to SAM background thread
                with _sam_lock:
                    _latest_rgb_for_sam[0] = rgb.copy()
                    _latest_det_for_sam[0] = {
                        k: (list(v) if isinstance(v,list) else v)
                        for k,v in self._live_detections.items()
                    }

            rgb_yolo    = draw_yolo_boxes(rgb, self._live_detections)
            depth_color = colorize_depth(depth)
            hud = build_capture_hud(rgb_yolo, depth_color,
                                    len(self.frames), self._last_flash,
                                    sam_panel=self._last_sam_panel)
            cv2.imshow("Panoramic Capture", hud)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # CAPTURE: YOLO → SAM → store
                print(f"  [Frame {len(self.frames)+1}] YOLO…", end='', flush=True)
                dets = self._run_yolo(rgb)
                print(" SAM…", end='', flush=True)
                dets = self._apply_sam(rgb, dets)
                n_sam = sum(
                    1
                    for grp in ('bed','bedding','pillows','obstacles','walls')
                    for obj in ([dets[grp]] if isinstance(dets[grp], dict) and dets[grp]
                                else (dets[grp] or []))
                    if obj.get('sam_mask') is not None
                )
                print(f" ✓  ({n_sam} SAM masks)")

                f = CapturedFrame(rgb=rgb.copy(), depth=depth.copy(),
                                  detections=dets, index=len(self.frames))
                self.frames.append(f)
                self._last_flash = time.time()
                bed_str  = "Bed ✓" if dets['bed'] else "NO BED"
                obj_list = (
                    ([dets['bed']['class']] if dets['bed'] else []) +
                    [o['class'] for o in dets['bedding']] +
                    [o['class'] for o in dets['pillows']] +
                    ([f"obstacle×{len(dets['obstacles'])}"] if dets['obstacles'] else []) +
                    ([f"Wall×{len(dets['walls'])}"] if dets['walls'] else [])
                )
                print(f"  Frame {len(self.frames):02d} — {bed_str} — {[x for x in obj_list if x]}")

                # ── Save SAM visualisation to disk ──────────
                try:
                    import os
                    sam_dir = Path("results/sam_frames"); sam_dir.mkdir(parents=True, exist_ok=True)
                    sam_vis = draw_sam_masks(rgb, dets, alpha=0.50)
                    # Draw class labels on mask
                    for grp in ('bed','bedding','pillows','obstacles','walls'):
                        val = dets.get(grp)
                        items = [val] if isinstance(val,dict) and val else (val or [])
                        for obj in items:
                            if obj.get('sam_mask') is None: continue
                            x1,y1,x2,y2 = [int(v) for v in obj['bbox']]
                            cls = obj.get('class','?')
                            cv2.putText(sam_vis, cls, (x1+2,y1+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                    sam_path = sam_dir / f"sam_frame_{len(self.frames):02d}.png"
                    cv2.imwrite(str(sam_path), sam_vis)
                        # Update live SAM panel with this captured frame's masks
                    self._last_sam_panel = sam_vis.copy()
                    print(f"      → saved {sam_path}")
                except Exception as e:
                    print(f"      [SAM save] {e}")

            elif key == ord('u') and self.frames:
                removed = self.frames.pop()
                print(f"  Undo — removed frame {removed.index + 1}")

            elif key in (ord('f'), ord('\r'), 13, 10):
                if len(self.frames) < 2:
                    print("  ⚠  Need at least 2 frames — keep capturing (SPACE)")
                    continue
                print(f"\n  Fusing {len(self.frames)} frames…")
                break

            elif key == ord('q'):
                self.cleanup()
                import sys; sys.exit(0)

        # Stop background SAM thread
        _stop_sam.set()

        # ── Fusion ────────────────────────────────────────
        bed_bbox = self._best_bed_bbox()
        if not bed_bbox:
            print("  ⚠  No bed detected — accessibility unknown")
            accessibility = {s: 'unknown' for s in ['head','foot','left','right']}
            stats = {}
        else:
            from semantic_depth_fusion import MultiViewFusion as SmartFusion
            smart = SmartFusion()
            accessibility, stats = smart.fuse(self.frames, bed_bbox)

        objects = aggregate_objects(self.frames)

        print("\n" + "─"*60)
        print("FUSION RESULT")
        print("─"*60)
        icons = {'free':'✓','partially_blocked':'⚠','blocked':'✗','unknown':'?'}
        for side in ['head','foot','left','right']:
            s  = accessibility.get(side, 'unknown')
            st = stats.get(side, {})
            print(f"  {side.upper():>5s}:  {icons[s]} {s:22s}"
                  f"  ({st.get('median_depth_m',0):.2f}m, "
                  f"conf={st.get('confidence',0):.2f}, "
                  f"views={st.get('num_views',0)})")

        names = (
            (['Bed'] if objects['bed'] else []) +
            [o['class'] for o in objects['bedding']] +
            [o['class'] for o in objects['pillows']] +
            (['obstacles'] if objects['obstacles'] else [])
        )
        print(f"\n  Objects detected: {names if names else 'none'}")

        result_panel = build_result_panel(accessibility, stats, objects, len(self.frames))
        cv2.imshow("Panoramic Capture", result_panel)
        print("\n[Auto-continuing in 3 seconds… or press any key now]")
        cv2.waitKey(3000)
        cv2.destroyWindow("Panoramic Capture")

        return accessibility, stats, objects, self.frames

    def cleanup(self):
        try:    self.pipeline.stop()
        except: pass
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────
def main():
    mapper = InteractivePanoramicMapper()
    try:
        accessibility, stats, objects, frames = mapper.run()
        out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {
            'timestamp': ts, 'num_frames': len(frames),
            'accessibility': accessibility, 'stats': stats,
            'detected_objects': {
                'bed_found': objects['bed'] is not None,
                'bedding':   [o['class'] for o in objects['bedding']],
                'pillows':   [o['class'] for o in objects['pillows']],
                'obstacles': len(objects['obstacles']),
            }
        }
        path = out_dir / f"panoramic_{ts}.json"
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ Saved: {path}")
    finally:
        mapper.cleanup()

if __name__ == "__main__":
    main()
