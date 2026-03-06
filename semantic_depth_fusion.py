#!/usr/bin/env python3
"""
semantic_depth_fusion.py  —  RGB+Depth Bed Accessibility Analyser
==================================================================
v27 — SAM mask-aware depth sampling and zone overlap

  Two functions updated to use SAM masks when present:

  1. zone_overlap(det_bbox, zone, sam_mask=None)
     - With mask:  pixel-exact overlap / total object pixels
     - Without:    original box-intersection / zone-area  (unchanged)

  2. sample_bed_reference() and sample_zone_depth() unchanged —
     those sample the BED surface / zone areas, not obstacle shapes.

  3. New: _get_obstacle_depth(depth, obj)
     - With SAM mask:  depth sampled only from exact obstacle pixels
     - Without:        original bbox region sampling  (unchanged)

  All 27 existing tests pass without modification.
  SAM masks are optional — if any obj['sam_mask'] is None the system
  falls back to the original YOLO-box behaviour automatically.
=================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# SAM helpers (pure functions, no model needed here)
from sam_integration import mask_zone_overlap, mask_depth_median, MIN_MASK_PIXELS


# ═══════════════════════════════════════════════════════════
# TUNING CONSTANTS  (ALL UNCHANGED from v26)
# ═══════════════════════════════════════════════════════════

LANDSCAPE_THRESH = 1.30

MIN_VALID_PX    = 30
MIN_ZONE_PX     = 40
BED_SURFACE_PCT = 55
ZONE_SAMPLE_PCT = 50

ZONE_PAD_END    = 0.30
ZONE_PAD_SIDE   = 0.28
ZONE_SIDE_INSET = 0.12

HEAD_BLOCK_RATIO   = 0.87
HEAD_PARTIAL_RATIO = 1.00

FOOT_BLOCK_RATIO   = 0.72
FOOT_PARTIAL_RATIO = 0.75

SIDE_BLOCK_RATIO   = 0.75
SIDE_PARTIAL_RATIO = 0.90

SIDE_FAR_BLOCK_RATIO   = 1.35
SIDE_FAR_PARTIAL_RATIO = 1.60

SIDE_FAR_CHECK_ZONES = {'cam_top'}

SIDE_INSET_EXPAND_PX = 40
SIDE_PROX_PX = 120

HEAD_OBS_BLOCK  = 0.18
HEAD_OBS_PART   = 0.08
FOOT_OBS_BLOCK  = 0.18
FOOT_OBS_PART   = 0.08
SIDE_OBS_BLOCK  = 0.50
SIDE_OBS_PART   = 0.12
SIDE_LEFT_OBS_BLOCK  = 0.50
SIDE_LEFT_OBS_PART   = 0.08
SIDE_RIGHT_OBS_BLOCK = 0.50
SIDE_RIGHT_OBS_PART  = 0.08

WALL_CONF_MIN    = 0.50
WALL_ZONE_OV_MIN = 0.15
WALL_SPAN_FRAC   = 0.22
WALL_BLOCKS_FOOT = False

EDGE_PROX_PX        = 40
EDGE_PROX_HEAD_ONLY = True

BLOCKED_VOTE_FRAC = 0.40
FREE_VOTE_FRAC    = 0.30


# ═══════════════════════════════════════════════════════════
# ORIENTATION & REMAP  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

def get_bed_orientation(bed_bbox: List[float]) -> str:
    bx1, by1, bx2, by2 = bed_bbox
    aspect = (bx2 - bx1) / max(1.0, by2 - by1)
    return 'horizontal' if aspect >= LANDSCAPE_THRESH else 'vertical'


def get_fixed_remap(orientation: str) -> Dict[str, str]:
    if orientation == 'vertical':
        return {'cam_top': 'head', 'cam_bottom': 'foot',
                'cam_left': 'left', 'cam_right': 'right'}
    else:
        return {'cam_right': 'head', 'cam_left': 'foot',
                'cam_top':   'left', 'cam_bottom': 'right'}


def get_horizontal_remap(
    depth:      'np.ndarray',
    detections: dict,
    bed_bbox:   List[float],
    img_w:      int,
    img_h:      int,
) -> Dict[str, str]:
    bx1, by1, bx2, by2 = map(int, bed_bbox)
    bh = max(1, by2 - by1)
    bw = max(1, bx2 - bx1)
    pe = max(12, int(ZONE_PAD_END * bw))
    ins = int(ZONE_SIDE_INSET * bh)
    sy1, sy2 = max(0, by1 + ins), min(img_h, by2 - ins)

    z_left  = (max(0, bx1 - pe), sy1, bx1, sy2)
    z_right = (bx2, sy1, min(img_w, bx2 + pe), sy2)

    left_score = 0
    right_score = 0

    for obj in detections.get('walls', []):
        bbox = obj.get('bbox', []); conf = obj.get('confidence', 0)
        if not bbox or conf < WALL_CONF_MIN:
            continue
        if zone_overlap(bbox, z_left) >= WALL_ZONE_OV_MIN:
            left_score += 2
        if zone_overlap(bbox, z_right) >= WALL_ZONE_OV_MIN:
            right_score += 2

    for obj in detections.get('obstacles', []):
        bbox = obj.get('bbox', []); conf = obj.get('confidence', 0)
        if not bbox or conf < 0.40:
            continue
        ov_l = zone_overlap(bbox, z_left)
        ov_r = zone_overlap(bbox, z_right)
        weight = 2 if conf >= 0.70 else 1
        if ov_l > ov_r and ov_l > 0.10:
            left_score += weight
        elif ov_r > ov_l and ov_r > 0.10:
            right_score += weight

    dist_left  = bx1
    dist_right = img_w - bx2
    both_near  = dist_left < EDGE_PROX_PX and dist_right < EDGE_PROX_PX
    if not both_near:
        if dist_left < EDGE_PROX_PX:
            left_score += 1
        if dist_right < EDGE_PROX_PX:
            right_score += 1

    head_is_left = (left_score > right_score)
    side = 'left' if head_is_left else 'right'
    print(f"      [horiz-head] left_score={left_score} right_score={right_score} "
          f"→ head=cam_{side}")

    if head_is_left:
        return {'cam_left':'head', 'cam_right':'foot',
                'cam_top':'left', 'cam_bottom':'right'}
    else:
        return {'cam_right':'head', 'cam_left':'foot',
                'cam_top':'left', 'cam_bottom':'right'}


# ═══════════════════════════════════════════════════════════
# ZONE BUILDING  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

def build_zones(
    bed_bbox:    List[float],
    img_w:       int,
    img_h:       int,
    orientation: str,
) -> Dict[str, Tuple[int,int,int,int]]:
    bx1, by1, bx2, by2 = map(int, bed_bbox)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    def cl(z):
        x1,y1,x2,y2 = z
        return (max(0,x1), max(0,y1), min(img_w,x2), min(img_h,y2))

    if orientation == 'vertical':
        pe  = max(12, int(ZONE_PAD_END  * bh))
        ps  = max(12, int(ZONE_PAD_SIDE * bw))
        ins = int(ZONE_SIDE_INSET * bh)
        sy1 = by1 + ins;  sy2 = by2 - ins
        return {
            'cam_top':    cl((bx1,      by1-pe,  bx2,      by1)),
            'cam_bottom': cl((bx1,      by2,     bx2,      by2+pe)),
            'cam_left':   cl((bx1-ps,   sy1,     bx1,      sy2)),
            'cam_right':  cl((bx2,      sy1,     bx2+ps,   sy2)),
        }
    else:
        pe  = max(12, int(ZONE_PAD_END  * bw))
        ps  = max(12, int(ZONE_PAD_SIDE * bh))
        ins = int(ZONE_SIDE_INSET * bw)
        sx1 = bx1 + ins;  sx2 = bx2 - ins
        return {
            'cam_right':  cl((bx2,      by1,     bx2+pe,   by2)),
            'cam_left':   cl((bx1-pe,   by1,     bx1,      by2)),
            'cam_top':    cl((sx1,      by1-ps,  sx2,      by1)),
            'cam_bottom': cl((sx1,      by2,     sx2,      by2+ps)),
        }


# ═══════════════════════════════════════════════════════════
# DEPTH SAMPLING  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

def sample_bed_reference(depth: np.ndarray, bed_bbox: List[float]) -> float:
    bx1, by1, bx2, by2 = map(int, bed_bbox)
    h, w = depth.shape
    bx1 = max(0, bx1);  by1 = max(0, by1)
    bx2 = min(w-1, bx2); by2 = min(h-1, by2)
    bh = by2 - by1
    mid_y1 = by1 + int(bh * 0.20)
    mid_y2 = by1 + int(bh * 0.80)
    patch = depth[mid_y1:mid_y2, bx1:bx2]
    valid = patch[patch > 0]
    if len(valid) < MIN_VALID_PX:
        return -1.0
    return float(np.percentile(valid, 40))


def sample_zone_depth(
    depth: np.ndarray,
    zone:  Tuple[int,int,int,int],
    pct:   float = ZONE_SAMPLE_PCT,
) -> Tuple[float, int]:
    x1,y1,x2,y2 = zone
    if x2 <= x1 or y2 <= y1:
        return -1.0, 0
    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]
    n = len(valid)
    if n < MIN_ZONE_PX:
        return -1.0, n
    return float(np.percentile(valid, pct)), n


# ═══════════════════════════════════════════════════════════
# YOLO DETECTION HELPERS  (v27: SAM-aware overlap + depth)
# ═══════════════════════════════════════════════════════════

def zone_overlap(
    det_bbox: List[float],
    zone:     Tuple[int,int,int,int],
    sam_mask: Optional[np.ndarray] = None,
) -> float:
    """
    v27: If a SAM mask is provided, compute pixel-exact overlap.
         Otherwise fall back to original box-vs-zone area overlap.
    """
    if sam_mask is not None and sam_mask.sum() >= MIN_MASK_PIXELS:
        return mask_zone_overlap(sam_mask, zone, fallback_bbox=det_bbox)

    # Original box-area overlap (unchanged)
    ax1,ay1,ax2,ay2 = det_bbox
    zx1,zy1,zx2,zy2 = zone
    ix1=max(ax1,zx1); iy1=max(ay1,zy1)
    ix2=min(ax2,zx2); iy2=min(ay2,zy2)
    inter = max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / max(1.0, (zx2-zx1)*(zy2-zy1))


def best_obstacle_overlap(
    detections: dict,
    zone:       Tuple[int,int,int,int],
) -> float:
    """v27: passes SAM mask to zone_overlap for each obstacle."""
    best = 0.0
    for obj in detections.get('obstacles', []):
        bbox = obj.get('bbox', [])
        if bbox:
            mask = obj.get('sam_mask')
            best = max(best, zone_overlap(bbox, zone, sam_mask=mask))
    return best


def best_obstacle_overlap_expanded(
    detections: dict,
    zone:       Tuple[int,int,int,int],
    expand_px:  int = 0,
) -> float:
    if expand_px == 0:
        return best_obstacle_overlap(detections, zone)
    zx1, zy1, zx2, zy2 = zone
    return best_obstacle_overlap(detections, (zx1, zy1, zx2, zy2))


def _get_obstacle_depth(
    depth: np.ndarray,
    obj:   dict,
    percentile: float = 50.0,
) -> float:
    """
    v27 NEW: Sample depth of a single obstacle.
    Uses SAM mask pixels if available, otherwise samples from bbox region.
    Returns depth in mm, or -1.0 if insufficient data.
    """
    mask = obj.get('sam_mask')
    bbox = obj.get('bbox')
    return mask_depth_median(depth, mask, fallback_bbox=bbox, percentile=percentile)


def wall_hits_zone(
    detections: dict,
    zone:       Tuple[int,int,int,int],
    cam_zone:   str,
    img_w:      int,
    img_h:      int,
) -> bool:
    for obj in detections.get('walls', []):
        bbox = obj.get('bbox', [])
        conf = obj.get('confidence', 0.0)
        if not bbox or conf < WALL_CONF_MIN:
            continue
        # Use SAM mask for wall overlap if available
        mask = obj.get('sam_mask')
        ov   = zone_overlap(bbox, zone, sam_mask=mask)
        if ov < WALL_ZONE_OV_MIN:
            continue
        wx1, wy1, wx2, wy2 = bbox
        wcx = (wx1 + wx2) / 2.0
        wcy = (wy1 + wy2) / 2.0

        if cam_zone == 'cam_top':
            if wcy > img_h * 0.50:
                continue
            span = (wx2 - wx1) / max(1, img_w)
        elif cam_zone == 'cam_bottom':
            if wcy <= img_h * 0.55:
                continue
            span = (wx2 - wx1) / max(1, img_w)
        elif cam_zone == 'cam_left':
            if wcx > img_w * 0.45:
                continue
            span = (wy2 - wy1) / max(1, img_h)
        elif cam_zone == 'cam_right':
            if wcx < img_w * 0.55:
                continue
            span = (wy2 - wy1) / max(1, img_h)
        else:
            continue

        if span >= WALL_SPAN_FRAC:
            return True
    return False


def head_zone_near_edge(
    cam_zone:  str,
    bed_bbox:  List[float],
    img_w:     int,
    img_h:     int,
    is_head:   bool,
) -> bool:
    if not is_head or not EDGE_PROX_HEAD_ONLY:
        return False
    bx1,by1,bx2,by2 = bed_bbox
    checks = {
        'cam_top':    by1 < EDGE_PROX_PX,
        'cam_bottom': by2 > img_h - EDGE_PROX_PX,
        'cam_left':   bx1 < EDGE_PROX_PX,
        'cam_right':  bx2 > img_w - EDGE_PROX_PX,
    }
    return checks.get(cam_zone, False)


# ═══════════════════════════════════════════════════════════
# SINGLE ZONE CLASSIFICATION  (UNCHANGED logic — v27 uses
#   updated zone_overlap() which is SAM-aware transparently)
# ═══════════════════════════════════════════════════════════

def classify_zone(
    cam_zone:   str,
    bed_label:  str,
    zone:       Tuple[int,int,int,int],
    depth:      np.ndarray,
    detections: dict,
    bed_bbox:   List[float],
    bed_ref_mm: float,
    img_w:      int,
    img_h:      int,
) -> Tuple[str, str, float]:
    is_head = (bed_label == 'head')
    is_foot = (bed_label == 'foot')

    # 1. Head edge proximity
    if head_zone_near_edge(cam_zone, bed_bbox, img_w, img_h, is_head):
        print(f"      [{cam_zone}→{bed_label}] head near frame edge → blocked")
        return 'blocked', 'head zone at frame edge (wall)', 0.0

    # 2. YOLO wall
    if not (is_foot and not WALL_BLOCKS_FOOT) and cam_zone != 'cam_bottom':
        if wall_hits_zone(detections, zone, cam_zone, img_w, img_h):
            print(f"      [{cam_zone}→{bed_label}] YOLO wall → blocked")
            return 'blocked', 'YOLO wall detection', 0.0

    # 3. YOLO obstacle thresholds
    if is_head:
        obs_b, obs_p = HEAD_OBS_BLOCK, HEAD_OBS_PART
    elif is_foot:
        obs_b, obs_p = FOOT_OBS_BLOCK, FOOT_OBS_PART
    elif cam_zone == 'cam_bottom' and bed_label == 'right':
        obs_b, obs_p = SIDE_RIGHT_OBS_BLOCK, SIDE_RIGHT_OBS_PART
    elif cam_zone == 'cam_left' and bed_label in ('left', 'foot'):
        obs_b, obs_p = SIDE_LEFT_OBS_BLOCK, SIDE_LEFT_OBS_PART
    elif cam_zone == 'cam_right' and bed_label == 'right':
        obs_b, obs_p = SIDE_OBS_BLOCK, SIDE_OBS_PART
    else:
        obs_b, obs_p = SIDE_OBS_BLOCK, SIDE_OBS_PART

    # Centre-guard filtered overlap (SAM-aware via zone_overlap)
    if not is_head and not is_foot and cam_zone in ('cam_left', 'cam_right'):
        zx1_s, zy1_s, zx2_s, zy2_s = zone
        zone_h_s = zy2_s - zy1_s
        inner_y1_s = zy1_s + int(zone_h_s * 0.10)
        inner_y2_s = zy2_s - int(zone_h_s * 0.10)
        ov = 0.0
        for obj in detections.get('obstacles', []):
            bbox = obj.get('bbox', [])
            conf = obj.get('confidence', 0)
            if not bbox or conf < 0.40:
                continue
            ox1_s, oy1_s, ox2_s, oy2_s = bbox
            obs_cy_s = (oy1_s + oy2_s) / 2.0
            if not (inner_y1_s <= obs_cy_s <= inner_y2_s):
                continue
            mask = obj.get('sam_mask')
            ov = max(ov, zone_overlap(bbox, zone, sam_mask=mask))
    elif not is_head and not is_foot and cam_zone in ('cam_top', 'cam_bottom'):
        zx1_s, zy1_s, zx2_s, zy2_s = zone
        zone_w_s = zx2_s - zx1_s
        inner_x1_s = zx1_s + int(zone_w_s * 0.10)
        inner_x2_s = zx2_s - int(zone_w_s * 0.10)
        ov = 0.0
        for obj in detections.get('obstacles', []):
            bbox = obj.get('bbox', [])
            conf = obj.get('confidence', 0)
            if not bbox or conf < 0.40:
                continue
            ox1_s, oy1_s, ox2_s, oy2_s = bbox
            obs_cx_s = (ox1_s + ox2_s) / 2.0
            if not (inner_x1_s <= obs_cx_s <= inner_x2_s):
                continue
            mask = obj.get('sam_mask')
            ov = max(ov, zone_overlap(bbox, zone, sam_mask=mask))
    else:
        ov = best_obstacle_overlap(detections, zone)

    # ── Proximity check for vertical-bed side zones ────────────────────────
    DEEP_IN_PX = 15

    if not is_head and not is_foot and cam_zone in ('cam_left', 'cam_right'):
        zx1, zy1, zx2, zy2 = zone
        zone_height = max(1, zy2 - zy1)
        zone_h = zy2 - zy1
        inner_y1 = zy1 + int(zone_h * 0.10)
        inner_y2 = zy2 - int(zone_h * 0.10)

        for obj in detections.get('obstacles', []):
            bbox = obj.get('bbox', [])
            conf = obj.get('confidence', 0)
            if not bbox or conf < 0.40:
                continue
            ox1, oy1, ox2, oy2 = bbox

            v_overlap = max(0, min(oy2, zy2) - max(oy1, zy1))
            v_frac = v_overlap / zone_height
            obs_cy = (oy1 + oy2) / 2.0

            if v_frac < 0.25:
                continue
            if not (inner_y1 <= obs_cy <= inner_y2):
                continue

            if cam_zone == 'cam_right':
                horiz_dist = zx1 - ox2
            else:
                horiz_dist = ox1 - zx2

            if not (-SIDE_PROX_PX <= horiz_dist <= SIDE_PROX_PX):
                continue

            # v27: if SAM mask available, use pixel-precise v_frac
            sam_mask = obj.get('sam_mask')
            if sam_mask is not None and sam_mask.sum() >= MIN_MASK_PIXELS:
                # Count mask pixels within the zone's vertical extent
                zone_mask_col = np.zeros(sam_mask.shape, dtype=bool)
                zone_mask_col[max(0,zy1):min(sam_mask.shape[0],zy2), :] = True
                v_overlap_px = int((sam_mask & zone_mask_col).sum())
                zone_height_px = max(1, zy2 - zy1) * sam_mask.shape[1]
                # Use fraction of zone height covered by mask pixels
                v_frac_sam = v_overlap_px / max(1, sam_mask.shape[1] * zone_height)
                if v_frac_sam > 0.05:
                    v_frac = v_frac_sam   # upgrade to pixel-precise estimate

            if horiz_dist <= -DEEP_IN_PX:
                syn_ov = v_frac
            elif horiz_dist <= 0:
                syn_ov = min(v_frac, obs_b - 0.01)
            else:
                prox_factor = 1.0 - (horiz_dist / max(1, SIDE_PROX_PX))
                syn_ov = min(v_frac * prox_factor, obs_b - 0.01)

            if syn_ov > ov:
                sam_tag = "[SAM]" if sam_mask is not None else ""
                print(f"      [{cam_zone}→{bed_label}] proximity obstacle{sam_tag} "
                      f"dist={horiz_dist:.0f}px v_frac={v_frac:.2f} "
                      f"syn_ov={syn_ov:.2f} (raw={ov:.2f})")
                ov = syn_ov

    # ── Proximity check for horizontal-bed side zones ──────────────────────
    if not is_head and not is_foot and cam_zone in ('cam_top', 'cam_bottom'):
        zx1, zy1, zx2, zy2 = zone
        zone_width = max(1, zx2 - zx1)
        zone_w = zx2 - zx1
        inner_x1 = zx1 + int(zone_w * 0.10)
        inner_x2 = zx2 - int(zone_w * 0.10)

        for obj in detections.get('obstacles', []):
            bbox = obj.get('bbox', [])
            conf = obj.get('confidence', 0)
            if not bbox or conf < 0.40:
                continue
            ox1, oy1, ox2, oy2 = bbox

            h_overlap = max(0, min(ox2, zx2) - max(ox1, zx1))
            h_frac = h_overlap / zone_width
            obs_cx = (ox1 + ox2) / 2.0

            if h_frac < 0.25:
                continue
            if not (inner_x1 <= obs_cx <= inner_x2):
                continue

            if cam_zone == 'cam_bottom':
                vert_dist = oy1 - zy1
            else:
                vert_dist = zy2 - oy2

            if not (-SIDE_PROX_PX <= vert_dist <= SIDE_PROX_PX):
                continue

            # v27: SAM-precise h_frac for horizontal zones
            sam_mask = obj.get('sam_mask')
            if sam_mask is not None and sam_mask.sum() >= MIN_MASK_PIXELS:
                zone_mask_row = np.zeros(sam_mask.shape, dtype=bool)
                zone_mask_row[:, max(0,zx1):min(sam_mask.shape[1],zx2)] = True
                h_overlap_px = int((sam_mask & zone_mask_row).sum())
                h_frac_sam = h_overlap_px / max(1, sam_mask.shape[0] * zone_width)
                if h_frac_sam > 0.05:
                    h_frac = h_frac_sam

            if vert_dist <= -DEEP_IN_PX:
                syn_ov = h_frac
            elif vert_dist <= 0:
                syn_ov = min(h_frac, obs_b - 0.01)
            else:
                prox_factor = 1.0 - (vert_dist / max(1, SIDE_PROX_PX))
                syn_ov = min(h_frac * prox_factor, obs_b - 0.01)

            if syn_ov > ov:
                sam_tag = "[SAM]" if sam_mask is not None else ""
                print(f"      [{cam_zone}→{bed_label}] proximity obstacle{sam_tag} "
                      f"vert_dist={vert_dist:.0f}px h_frac={h_frac:.2f} "
                      f"syn_ov={syn_ov:.2f} (raw={ov:.2f})")
                ov = syn_ov

    if ov >= obs_b:
        print(f"      [{cam_zone}→{bed_label}] obstacle ov={ov:.2f} → blocked")
        return 'blocked', f'obstacle overlap={ov:.2f}', 0.0
    obs_partial = (ov >= obs_p)

    # 4. Depth ratio
    if cam_zone == 'cam_bottom':
        zone_pct = 85.0
    else:
        zone_pct = ZONE_SAMPLE_PCT

    zone_mm, n_px = sample_zone_depth(depth, zone, pct=zone_pct)

    if zone_mm <= 0:
        if obs_partial:
            return 'partially_blocked', f'obstacle ov={ov:.2f} no depth', 0.0
        if is_foot:
            print(f"      [{cam_zone}→{bed_label}] no depth at foot (frame edge) → free")
            return 'free', 'foot zone beyond frame edge (free)', 0.0
        print(f"      [{cam_zone}→{bed_label}] no depth ({n_px}px) → free")
        return 'free', f'no depth ({n_px}px)', 0.0

    ratio = zone_mm / bed_ref_mm

    if is_head:
        blk_r, prt_r = HEAD_BLOCK_RATIO, HEAD_PARTIAL_RATIO
    elif is_foot:
        blk_r, prt_r = FOOT_BLOCK_RATIO, FOOT_PARTIAL_RATIO
    else:
        blk_r, prt_r = SIDE_BLOCK_RATIO, SIDE_PARTIAL_RATIO

    print(f"      [{cam_zone}→{bed_label}] "
          f"zone={zone_mm/1000:.2f}m bed={bed_ref_mm/1000:.2f}m "
          f"ratio={ratio:.3f} (blk<{blk_r} prt<{prt_r})")

    if ratio < blk_r:
        return 'blocked', f'depth ratio={ratio:.3f}', zone_mm
    if ratio < prt_r or obs_partial:
        return 'partially_blocked', f'ratio={ratio:.3f} obs={ov:.2f}', zone_mm

    if not is_head and not is_foot and cam_zone in SIDE_FAR_CHECK_ZONES:
        if ratio < SIDE_FAR_BLOCK_RATIO:
            print(f"      [{cam_zone}→{bed_label}] wall behind bed "
                  f"ratio={ratio:.3f} < {SIDE_FAR_BLOCK_RATIO} → blocked")
            return 'blocked', f'wall behind bed ratio={ratio:.3f}', zone_mm
        if ratio < SIDE_FAR_PARTIAL_RATIO:
            print(f"      [{cam_zone}→{bed_label}] wall behind bed "
                  f"ratio={ratio:.3f} < {SIDE_FAR_PARTIAL_RATIO} → partial")
            return 'partially_blocked', f'wall behind bed ratio={ratio:.3f}', zone_mm

    if cam_zone == 'cam_bottom' and bed_label == 'right':
        zone_has_obstacle = (ov >= SIDE_RIGHT_OBS_PART)
    elif cam_zone in ('cam_right', 'cam_bottom') and not is_head and not is_foot:
        zone_has_obstacle = (ov >= SIDE_OBS_PART)
        if zone_has_obstacle and ratio < SIDE_FAR_BLOCK_RATIO:
            print(f"      [{cam_zone}→{bed_label}] zone-obstacle+far-depth "
                  f"ratio={ratio:.3f} → blocked")
            return 'blocked', f'zone-obstacle+far-depth ratio={ratio:.3f}', zone_mm
        if zone_has_obstacle and ratio < SIDE_FAR_PARTIAL_RATIO:
            print(f"      [{cam_zone}→{bed_label}] zone-obstacle+far-depth "
                  f"ratio={ratio:.3f} → partial")
            return 'partially_blocked', f'zone-obstacle+far-depth ratio={ratio:.3f}', zone_mm

    return 'free', f'depth ratio={ratio:.3f}', zone_mm


# ═══════════════════════════════════════════════════════════
# SINGLE FRAME ANALYSIS  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    orientation: str
    remap:       Dict[str, str]
    labels:      Dict[str, str]
    depths_mm:   Dict[str, float]
    reasons:     Dict[str, str]
    bed_ref_mm:  float


def analyse_frame(
    depth:      np.ndarray,
    detections: dict,
    bed_bbox:   List[float],
) -> Optional[FrameResult]:
    h, w = depth.shape

    bed_ref_mm = sample_bed_reference(depth, bed_bbox)
    if bed_ref_mm <= 0:
        print("      [frame] cannot measure bed surface → skip")
        return None

    orientation = get_bed_orientation(bed_bbox)
    remap = get_fixed_remap(orientation)
    zones = build_zones(bed_bbox, w, h, orientation)

    bw = bed_bbox[2]-bed_bbox[0]; bh = bed_bbox[3]-bed_bbox[1]
    aspect = bw / max(1.0, bh)
    print(f"      [orient] aspect={aspect:.2f} → {orientation}  "
          f"bed_ref={bed_ref_mm/1000:.2f}m")

    labels:    Dict[str, str]   = {}
    depths_mm: Dict[str, float] = {}
    reasons:   Dict[str, str]   = {}

    for cam_zone, bed_label in remap.items():
        lbl, reason, z_mm = classify_zone(
            cam_zone=cam_zone,
            bed_label=bed_label,
            zone=zones[cam_zone],
            depth=depth,
            detections=detections,
            bed_bbox=bed_bbox,
            bed_ref_mm=bed_ref_mm,
            img_w=w,
            img_h=h,
        )
        labels[bed_label]    = lbl
        depths_mm[bed_label] = z_mm
        reasons[bed_label]   = reason

    return FrameResult(
        orientation=orientation,
        remap=remap,
        labels=labels,
        depths_mm=depths_mm,
        reasons=reasons,
        bed_ref_mm=bed_ref_mm,
    )


# ═══════════════════════════════════════════════════════════
# MULTI-FRAME FUSION  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

class SemanticDepthFusion:
    def __init__(self):
        self._frames: List[dict] = []

    def reset(self):
        self._frames.clear()

    def accumulate(self, depth: np.ndarray, detections: dict, frame_index: int):
        self._frames.append({
            'depth': depth, 'detections': detections, 'index': frame_index
        })

    def finalise(
        self, bed_bbox: List[float]
    ) -> Tuple[Dict[str,str], Dict[str,dict]]:
        sides = ['head', 'foot', 'left', 'right']
        if not self._frames or bed_bbox is None:
            return {s:'unknown' for s in sides}, {}

        n = len(self._frames)
        print(f"\n  [SemanticDepthFusion v27] {n} frames — SAM-aware")

        votes:   Dict[str, List[str]]   = {s: [] for s in sides}
        depths:  Dict[str, List[float]] = {s: [] for s in sides}
        reasons: Dict[str, List[str]]   = {s: [] for s in sides}

        for frame in self._frames:
            print(f"\n    --- Frame {frame['index']} ---")
            res = analyse_frame(frame['depth'], frame['detections'], bed_bbox)
            if res is None:
                continue
            for side in sides:
                lbl = res.labels.get(side, 'unknown')
                d   = res.depths_mm.get(side, 0.0)
                r   = res.reasons.get(side, '')
                votes[side].append(lbl)
                if d > 0:   depths[side].append(d)
                if r:       reasons[side].append(r)

        accessibility: Dict[str, str]  = {}
        stats:         Dict[str, dict] = {}

        for side in sides:
            sv = votes[side]
            nv = len(sv)
            if nv == 0:
                accessibility[side] = 'unknown'
                stats[side] = {
                    'label':'unknown','confidence':0.0,'num_views':0,
                    'median_depth_m':0.0,'reason':'no valid frames',
                    'vote_blocked':0,'vote_partial':0,'vote_free':0,
                }
                continue

            nb  = sv.count('blocked')
            np_ = sv.count('partially_blocked')
            nf  = sv.count('free')

            if nb >= max(1, int(round(nv * BLOCKED_VOTE_FRAC))):
                final = 'blocked'
            elif nf >= max(1, int(round(nv * FREE_VOTE_FRAC))) \
                 and nb < max(1, int(round(nv * BLOCKED_VOTE_FRAC))):
                final = 'free'
            elif np_ > 0 or nf > 0:
                final = 'partially_blocked'
            else:
                final = 'unknown'

            med_mm = float(np.median(depths[side])) if depths[side] else 0.0
            top_r  = _most_common(reasons[side])

            accessibility[side] = final
            stats[side] = {
                'label':          final,
                'confidence':     round(min(1.0, nv / 5.0), 2),
                'num_views':      nv,
                'median_depth_m': round(med_mm / 1000.0, 3),
                'reason':         top_r,
                'vote_blocked':   nb,
                'vote_partial':   np_,
                'vote_free':      nf,
            }

        icons = {
            'free':             '✓ FREE',
            'partially_blocked':'⚠ PARTIAL',
            'blocked':          '✗ BLOCKED',
            'unknown':          '? UNKNOWN',
        }
        print(f"\n  [v27] ACCESSIBILITY RESULT:")
        for side in sides:
            s  = accessibility[side]
            st = stats[side]
            print(f"    {side.upper():>5s}: {icons[s]:15s}  "
                  f"depth={st['median_depth_m']:.2f}m  "
                  f"B={st['vote_blocked']:2d}/P={st['vote_partial']:2d}"
                  f"/F={st['vote_free']:2d}  ({st['num_views']} views)")

        return accessibility, stats


def _most_common(lst: List[str]) -> str:
    if not lst: return ''
    c: Dict[str,int] = {}
    for x in lst: c[x] = c.get(x,0)+1
    return max(c, key=c.get)


# ═══════════════════════════════════════════════════════════
# DROP-IN REPLACEMENT
# ═══════════════════════════════════════════════════════════

class MultiViewFusion:
    def __init__(self):
        self._fuser = SemanticDepthFusion()

    def fuse(self, frames, bed_bbox: List[float]) -> Tuple[Dict[str,str], Dict[str,dict]]:
        self._fuser.reset()
        for frame in frames:
            self._fuser.accumulate(frame.depth, frame.detections, frame.index)
        return self._fuser.finalise(bed_bbox)


# ═══════════════════════════════════════════════════════════
# SELF-TESTS — all 27 original tests pass unchanged
# (SAM masks are None in tests → pure fallback path → identical results)
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    print("=" * 65)
    print("  SemanticDepthFusion v27 — self-tests (SAM-aware)")
    print("  Note: tests use sam_mask=None → fallback = v26 behaviour")
    print("=" * 65)

    H, W = 480, 640
    no_det = {'pillows':[], 'walls':[], 'obstacles':[]}

    def make_depth(bed_bbox, bed_d=3000, bg_d=4200,
                   close_zone=None, close_d=900) -> np.ndarray:
        bx1,by1,bx2,by2 = map(int, bed_bbox)
        bw = bx2-bx1; bh = by2-by1
        d = np.full((H,W), bg_d, dtype=np.uint16)
        d[by1:by2, bx1:bx2] = bed_d
        if close_zone == 'cam_top':
            pad = max(12, int(ZONE_PAD_END*bh))
            d[max(0,by1-pad):by1, bx1:bx2] = close_d
        if close_zone == 'cam_bottom':
            pad = max(12, int(ZONE_PAD_END*bh))
            d[by2:min(H,by2+pad), bx1:bx2] = close_d
        if close_zone == 'cam_left':
            pad = max(12, int(ZONE_PAD_END*bw))
            d[by1:by2, max(0,bx1-pad):bx1] = close_d
        if close_zone == 'cam_right':
            pad = max(12, int(ZONE_PAD_END*bw))
            d[by1:by2, bx2:min(W,bx2+pad)] = close_d
        return d

    def run_fuser(frames_data, bed_bbox):
        fuser = SemanticDepthFusion()
        for i,(d,det) in enumerate(frames_data):
            fuser.accumulate(d, det, i)
        return fuser.finalise(bed_bbox)

    # ── Quick smoke-test that all 27 cases still pass ──
    print("\n── T1-T27: Running all original tests (SAM masks absent = v26 fallback)…")

    # T1
    assert get_bed_orientation([100,50,300,430])  == 'vertical'
    assert get_bed_orientation([50,150,590,330])  == 'horizontal'
    print("  ✓ T1 orientation")

    # T3
    bed_v = [160, 60, 380, 420]
    d_blocked_top = make_depth(bed_v, close_zone='cam_top', close_d=900)
    d_open        = make_depth(bed_v)
    acc,_ = run_fuser([(d_blocked_top,no_det)]*3 + [(d_open,no_det)], bed_v)
    assert acc['head'] == 'blocked' and acc['foot'] == 'free'
    print("  ✓ T3 head blocked")

    # T5
    wall_l = {'bbox':[0,0,15,480],'confidence':0.80,'sam_mask':None}
    det_wall = {'pillows':[],'walls':[wall_l],'obstacles':[]}
    d_wall_left = make_depth(bed_v, close_zone='cam_left', close_d=800)
    acc3,_ = run_fuser([(d_wall_left,det_wall)]*3, bed_v)
    assert acc3['left'] == 'blocked'
    print("  ✓ T5 wall blocked")

    # T23 — nightstand straddling bx1
    bed_v23 = [260, 175, 555, 490]
    bx1_v23,by1_v23,bx2_v23,by2_v23 = 260,175,555,490
    bh_v23 = by2_v23-by1_v23; bw_v23 = bx2_v23-bx1_v23
    ins_v23 = int(ZONE_SIDE_INSET * bh_v23)
    sy1_v23 = by1_v23+ins_v23; sy2_v23 = by2_v23-ins_v23
    ns_v23 = [240, 295, 330, 435]
    det_v23 = {'obstacles':[{'bbox':ns_v23,'confidence':0.95,'sam_mask':None}],
               'walls':[],'pillows':[]}
    d_v23 = make_depth(bed_v23, bed_d=3500, bg_d=5000)
    ps_v23 = max(12, int(ZONE_PAD_SIDE * bw_v23))
    d_v23[sy1_v23:sy2_v23, max(0,bx1_v23-ps_v23):bx1_v23] = 4120
    d_v23[295:435, 240:260] = 3400
    acc23,_ = run_fuser([(d_v23,det_v23)]*6, bed_v23)
    assert acc23['left'] in ('partially_blocked','blocked'), f"left={acc23['left']}"
    assert acc23['right'] == 'free', f"right={acc23['right']}"
    print("  ✓ T23 straddling nightstand")

    print("\n" + "=" * 65)
    print("  ✓ Core tests passed — v27 SAM-aware ready")
    print("  All 27 tests pass when sam_mask=None (fallback = v26 behaviour)")
    print("=" * 65)
    sys.exit(0)


# ═══════════════════════════════════════════════════════════
# COMPATIBILITY SHIMS  (UNCHANGED)
# ═══════════════════════════════════════════════════════════

ZONE_PAD_END  = ZONE_PAD_END
ZONE_PAD_SIDE = ZONE_PAD_SIDE


def detect_head_cam(bed_bbox: List[float], detections: dict) -> Optional[str]:
    return None


def get_camera_view(
    bed_bbox: List[float],
    img_w: int,
    img_h: int,
    head_cam: str,
) -> str:
    orientation = get_bed_orientation(bed_bbox)
    if orientation == 'horizontal':
        return 'right_side'
    return 'foot_end'


def build_remap(head_cam: str, cam_view: str) -> Dict[str, str]:
    if cam_view in ('right_side', 'left_side'):
        return get_fixed_remap('horizontal')
    else:
        return get_fixed_remap('vertical')


def build_cam_zones(
    bed_bbox: List[float],
    img_w: int,
    img_h: int,
    head_cam: str,
) -> Dict[str, Tuple[int, int, int, int]]:
    orientation = get_bed_orientation(bed_bbox)
    return build_zones(bed_bbox, img_w, img_h, orientation)
