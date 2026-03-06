#!/usr/bin/env python3
"""
sam_integration.py  —  SAM mask refinement for LEO bed-making robot
=====================================================================
v4 — FastSAM backend for real CPU performance

Why FastSAM instead of MobileSAM/ViT-B:
  SAM ViT-B    ~1900ms/frame  — ViT transformer, too slow
  MobileSAM    ~1900ms/frame  — still ViT-based, AVX2 CPU bottleneck
  FastSAM-s    ~200-400ms/frame  — YOLO CNN backbone, no transformer ← THIS

FastSAM uses a YOLO-v8 segmentation backbone instead of a Vision
Transformer encoder. CNN convolutions are much more CPU-friendly than
attention layers, giving 5-10x real speedup on Ryzen without AVX-512.

Install:
    pip install fastsam

Checkpoint (~23MB):
    wget -O models/sam/FastSAM-s.pt \
      "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM-s.pt"

API is 100% identical to v1/v2/v3 — no changes needed anywhere else.

    refiner = SAMRefiner(model_type="vit_b", device="cpu")  # same call
    detections = refiner.refine_batch(rgb, detections)       # same call
    vis = draw_sam_masks(rgb, detections, alpha=0.40)        # same call
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import torch

# ── Use all 8 Ryzen cores ─────────────────────────────────────────────
_N_THREADS = int(os.environ.get("SAM_NUM_THREADS", 8))
torch.set_num_threads(_N_THREADS)
try:
    torch.set_num_interop_threads(max(1, _N_THREADS // 2))
except RuntimeError:
    pass  # already set — safe to ignore

# ── FastSAM inference image size (smaller = faster, 512 is sweet spot) ─
FASTSAM_IMGSZ = int(os.environ.get("FASTSAM_IMGSZ", 512))

# ── IOU / confidence thresholds for FastSAM ───────────────────────────
FASTSAM_CONF  = float(os.environ.get("FASTSAM_CONF",  "0.4"))
FASTSAM_IOU   = float(os.environ.get("FASTSAM_IOU",   "0.9"))

# ── Checkpoint locations (searched in order) ─────────────────────────
_FASTSAM_CKPTS = [
    os.environ.get("FASTSAM_CKPT", ""),
    "models/sam/FastSAM-s.pt",
    "models/sam/FastSAM-x.pt",
    str(Path.home() / ".cache" / "fastsam" / "FastSAM-s.pt"),
]
_MOBILE_SAM_CKPTS = [
    os.environ.get("MOBILE_SAM_CKPT", ""),
    "models/sam/mobile_sam.pt",
]
_VITB_CKPTS = [
    "models/sam/sam_vit_b_01ec64.pth",
    os.environ.get("SAM_CKPT", ""),
]

MIN_MASK_PIXELS = 200

# Mask overlay colours (BGR, keyed by class name)
_MASK_COLORS = {
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
# SAMRefiner
# ─────────────────────────────────────────────────────────

class SAMRefiner:
    """
    Backend priority: FastSAM → MobileSAM → SAM ViT-B

    Constructor signature UNCHANGED from v1/v2/v3:
        refiner = SAMRefiner(model_type="vit_b", device="cpu")
    """

    def __init__(
        self,
        model_type: str = "vit_b",   # kept for API compat — ignored
        device: str = "cpu",
    ):
        self._device  = device
        self._backend = "none"
        self._model   = None       # FastSAM model (ultralytics)
        self._predictor = None     # SAM-style predictor (MobileSAM / ViT-B)
        self._current_image_id    = None
        self._current_image_shape = None
        self._last_full_masks     = None  # cached FastSAM result for frame

        print(f"\n[SAM] Initialising  threads={_N_THREADS}  fastsam_imgsz={FASTSAM_IMGSZ}  device={device}")

        if self._try_load_fastsam(device):
            return
        if self._try_load_mobile_sam(device):
            return
        if self._try_load_vitb(device):
            return

        raise RuntimeError(
            "[SAM] No backend could be loaded.\n"
            "  Recommended: pip install fastsam\n"
            "  Checkpoint:  wget -O models/sam/FastSAM-s.pt \\\n"
            "    'https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM-s.pt'"
        )

    # ── Loaders ───────────────────────────────────────────

    def _try_load_fastsam(self, device: str) -> bool:
        # The 'fastsam' pip package is a 1KB stub — real FastSAM lives in ultralytics
        try:
            from ultralytics import FastSAM as _FastSAM
            FastSAM = _FastSAM
        except (ImportError, AttributeError):
            print("[SAM] FastSAM not found in ultralytics — trying stub fastsam package")
            try:
                from fastsam import FastSAM
            except ImportError:
                print("[SAM] FastSAM unavailable. Install: pip install ultralytics")
                return False

        ckpt = next((p for p in _FASTSAM_CKPTS if p and Path(p).exists()), None)
        if ckpt is None:
            print("[SAM] FastSAM checkpoint not found.")
            print("      Expected at: models/sam/FastSAM-s.pt")
            print("      Download:  python -c \"from ultralytics import FastSAM; FastSAM('FastSAM-s.pt')\"")
            print("      Then:      mv FastSAM-s.pt models/sam/FastSAM-s.pt")
            return False

        try:
            self._model   = FastSAM(str(ckpt))
            self._backend = "fastsam"
            print(f"[SAM] ✓ FastSAM loaded  ({ckpt})")
            print(f"[SAM]   Expected speed: ~200-400ms/frame on Ryzen 7000 CPU")
            return True
        except Exception as e:
            print(f"[SAM] FastSAM load error: {e}")
            return False

    def _try_load_mobile_sam(self, device: str) -> bool:
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            ckpt = next((p for p in _MOBILE_SAM_CKPTS if p and Path(p).exists()), None)
            if ckpt is None:
                return False
            model = sam_model_registry["vit_t"](checkpoint=str(ckpt))
            model.to(device=device); model.eval()
            self._predictor = SamPredictor(model)
            self._backend   = "mobile_sam"
            print(f"[SAM] ✓ MobileSAM loaded  ({ckpt})  [~1900ms/frame — install fastsam for 5x speedup]")
            return True
        except Exception as e:
            print(f"[SAM] MobileSAM load error: {e}")
            return False

    def _try_load_vitb(self, device: str) -> bool:
        try:
            from segment_anything import SamPredictor, build_sam_vit_b
            ckpt = next((p for p in _VITB_CKPTS if p and Path(p).exists()), None)
            if ckpt is None:
                return False
            model = build_sam_vit_b(checkpoint=str(ckpt))
            model.to(device=device)
            self._predictor = SamPredictor(model)
            self._backend   = "vit_b"
            print(f"[SAM] ✓ SAM ViT-B loaded  ({ckpt})  [~1900ms/frame — install fastsam for 5x speedup]")
            return True
        except Exception as e:
            print(f"[SAM] SAM ViT-B load error: {e}")
            return False

    # ── Public API (unchanged from v1/v2/v3) ──────────────

    def refine(
        self,
        image_bgr: np.ndarray,
        bbox: list,
    ) -> Optional[np.ndarray]:
        """Single bbox refinement. Returns H×W bool mask or None."""
        try:
            if self._backend == "fastsam":
                return self._fastsam_single(image_bgr, bbox)
            else:
                self._set_image_sam(image_bgr)
                return self._predict_mask_sam(bbox)
        except Exception as e:
            print(f"[SAM] refine() failed: {e}")
            return None

    def refine_batch(
        self,
        image_bgr: np.ndarray,
        detections: dict,
    ) -> dict:
        """
        Adds 'sam_mask' (H×W bool | None) to every detection.
        For FastSAM: runs inference ONCE per frame, matches masks to boxes.
        For ViT backends: embeds image ONCE, predicts per box.
        API unchanged from v1/v2/v3.
        """
        try:
            if self._backend == "fastsam":
                return self._fastsam_batch(image_bgr, detections)
            else:
                self._set_image_sam(image_bgr)
                for key in ('bed', 'bedding', 'pillows', 'obstacles', 'walls'):
                    val = detections.get(key)
                    if val is None:
                        continue
                    if isinstance(val, list):
                        for obj in val:
                            obj['sam_mask'] = self._predict_mask_sam(obj['bbox'])
                    elif isinstance(val, dict):
                        val['sam_mask'] = self._predict_mask_sam(val['bbox'])
                return detections
        except Exception as e:
            print(f"[SAM] refine_batch() failed: {e}")
            return detections

    # ── FastSAM internals ─────────────────────────────────

    def _fastsam_run(self, image_bgr: np.ndarray):
        """
        Run FastSAM on the full image and cache all segment masks.
        Called once per new frame — subsequent bbox lookups reuse the cache.
        """
        img_id = id(image_bgr)
        if img_id == self._current_image_id:
            return  # already processed this frame

        everything = self._model(
            image_bgr,
            device=self._device,
            retina_masks=True,
            imgsz=FASTSAM_IMGSZ,
            conf=FASTSAM_CONF,
            iou=FASTSAM_IOU,
            verbose=False,
        )
        # Extract all masks as H×W bool arrays
        masks = []
        if everything and everything[0].masks is not None:
            for m in everything[0].masks.data:
                mask = m.cpu().numpy().astype(bool)
                # Resize to original image size if needed
                if mask.shape != image_bgr.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (image_bgr.shape[1], image_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                masks.append(mask)

        self._last_full_masks  = masks
        self._current_image_id = img_id

    def _best_mask_for_bbox(self, bbox: list, orig_h: int, orig_w: int) -> Optional[np.ndarray]:
        """
        From the cached FastSAM masks, pick the one with highest IoU
        against the YOLO bounding box.
        """
        if not self._last_full_masks:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(orig_w, x2); y2 = min(orig_h, y2)
        box_area = max(1, (x2 - x1) * (y2 - y1))

        # Build a boolean box mask for IoU computation
        box_mask = np.zeros((orig_h, orig_w), dtype=bool)
        box_mask[y1:y2, x1:x2] = True

        best_mask  = None
        best_score = 0.0

        for mask in self._last_full_masks:
            if mask.shape != (orig_h, orig_w):
                continue
            intersection = int((mask & box_mask).sum())
            if intersection == 0:
                continue
            union = int((mask | box_mask).sum())
            iou   = intersection / max(union, 1)
            # Also weight by how well the mask is contained in the box
            containment = intersection / max(int(mask.sum()), 1)
            score = 0.6 * iou + 0.4 * containment
            if score > best_score:
                best_score = score
                best_mask  = mask

        if best_mask is None or best_mask.sum() < MIN_MASK_PIXELS:
            return None
        return best_mask

    def _fastsam_single(self, image_bgr: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        self._fastsam_run(image_bgr)
        return self._best_mask_for_bbox(bbox, image_bgr.shape[0], image_bgr.shape[1])

    def _fastsam_batch(self, image_bgr: np.ndarray, detections: dict) -> dict:
        """Run FastSAM once, match masks to all YOLO boxes."""
        self._fastsam_run(image_bgr)
        orig_h, orig_w = image_bgr.shape[:2]
        for key in ('bed', 'bedding', 'pillows', 'obstacles', 'walls'):
            val = detections.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                for obj in val:
                    obj['sam_mask'] = self._best_mask_for_bbox(obj['bbox'], orig_h, orig_w)
            elif isinstance(val, dict):
                val['sam_mask'] = self._best_mask_for_bbox(val['bbox'], orig_h, orig_w)
        return detections

    # ── ViT-backend internals (MobileSAM / ViT-B) ─────────

    def _set_image_sam(self, image_bgr: np.ndarray):
        img_id = id(image_bgr)
        if img_id == self._current_image_id:
            return
        orig_h, orig_w = image_bgr.shape[:2]
        # Downscale to 512px for speed
        scale = 512 / max(orig_h, orig_w)
        if scale < 0.99:
            small = cv2.resize(image_bgr,
                               (int(orig_w * scale), int(orig_h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = image_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        self._current_image_id    = img_id
        self._current_image_shape = (orig_h, orig_w, scale)

    def _predict_mask_sam(self, bbox: list) -> Optional[np.ndarray]:
        if self._current_image_shape is None:
            return None
        orig_h, orig_w, scale = self._current_image_shape
        x1, y1, x2, y2 = bbox
        sb = np.array([x1*scale, y1*scale, x2*scale, y2*scale], dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            point_coords=None, point_labels=None,
            box=sb[None, :], multimask_output=True,
        )
        best_mask = masks[int(np.argmax(scores))].astype(np.uint8)
        if scale < 0.99:
            best_mask = cv2.resize(best_mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)
        bool_mask = best_mask.astype(bool)
        return bool_mask if bool_mask.sum() >= MIN_MASK_PIXELS else None


# ─────────────────────────────────────────────────────────
# Standalone helper functions
# (used by semantic_depth_fusion.py — API unchanged from v1/v2)
# ─────────────────────────────────────────────────────────

def mask_depth_median(
    depth: np.ndarray,
    mask: Optional[np.ndarray],
    fallback_bbox: Optional[list] = None,
    percentile: float = 50.0,
) -> float:
    """
    Sample depth values at SAM mask pixels and return the given percentile.
    Falls back to the bounding-box region if mask is None.
    Returns -1.0 if not enough valid pixels.
    """
    if mask is not None and mask.sum() >= MIN_MASK_PIXELS:
        valid = depth[mask]
        valid = valid[valid > 0]
        if len(valid) >= 30:
            return float(np.percentile(valid, percentile))

    if fallback_bbox is not None:
        x1, y1, x2, y2 = map(int, fallback_bbox)
        h, w = depth.shape
        region = depth[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        valid  = region[region > 0]
        if len(valid) >= 30:
            return float(np.percentile(valid, percentile))

    return -1.0


def mask_zone_overlap(
    mask: Optional[np.ndarray],
    zone: tuple,
    fallback_bbox: Optional[list] = None,
    img_h: int = 480,
    img_w: int = 640,
) -> float:
    """
    Compute fraction of the object overlapping the zone.
    With SAM mask → pixel-exact.  Without → box intersection (old behaviour).
    zone: (x1, y1, x2, y2) in pixels.  Returns ratio in [0, 1].
    """
    zx1, zy1, zx2, zy2 = zone

    if mask is not None and mask.sum() >= MIN_MASK_PIXELS:
        zone_mask = np.zeros(mask.shape, dtype=bool)
        zone_mask[
            max(0, zy1):min(mask.shape[0], zy2),
            max(0, zx1):min(mask.shape[1], zx2)
        ] = True
        total_px = int(mask.sum())
        if total_px == 0:
            return 0.0
        return int((mask & zone_mask).sum()) / total_px

    if fallback_bbox is None:
        return 0.0

    ax1, ay1, ax2, ay2 = fallback_bbox
    ix1 = max(ax1, zx1); iy1 = max(ay1, zy1)
    ix2 = min(ax2, zx2); iy2 = min(ay2, zy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / max(1.0, (zx2 - zx1) * (zy2 - zy1))


def draw_sam_masks(
    image_bgr: np.ndarray,
    detections: dict,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay SAM masks on the image for visualisation.
    Called from draw_yolo_boxes() in depth_mapper.py.
    API unchanged from v1/v2.
    """
    vis     = image_bgr.copy()
    overlay = vis.copy()

    for key in ('bed', 'bedding', 'pillows', 'obstacles', 'walls'):
        val = detections.get(key)
        if not val:
            continue
        items = [val] if isinstance(val, dict) else val
        for obj in items:
            mask = obj.get('sam_mask')
            if mask is None:
                continue
            color = _MASK_COLORS.get(obj.get('class', 'unknown'), (200, 200, 200))
            overlay[mask] = color

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis
