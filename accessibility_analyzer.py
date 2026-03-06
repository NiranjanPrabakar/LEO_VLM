import numpy as np
import cv2

class AccessibilityAnalyzer:

    def __init__(self,
                 wall_threshold_offset=700,
                 free_threshold_offset=250,
                 blocked_percent_threshold=0.65,
                 free_percent_threshold=0.25):

        self.wall_offset = wall_threshold_offset
        self.free_offset = free_threshold_offset
        self.blocked_percent = blocked_percent_threshold
        self.free_percent = free_percent_threshold

    # ------------------------------------------------------------
    def analyze_bed_accessibility(self, depth, bed_bbox, obstacles=None):
        """
        depth : depth map (H x W) in mm
        bed_bbox : [x1, y1, x2, y2]
        obstacles : list of obstacle bboxes [[x1, y1, x2, y2], ...]
        """
        x1, y1, x2, y2 = map(int, bed_bbox)

        h, w = depth.shape
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)

        bed_region = depth[y1:y2, x1:x2]

        valid_bed = bed_region[bed_region > 0]
        if valid_bed.size < 100:
            return None, None

        bed_depth = np.median(valid_bed)

        pad_x = int(0.25 * (x2 - x1))
        pad_y = int(0.25 * (y2 - y1))

        regions = {
            "left": depth[y1:y2, max(0, x1 - pad_x):x1],
            "right": depth[y1:y2, x2:min(w, x2 + pad_x)],
            "head": depth[max(0, y1 - pad_y):y1, x1:x2],
            "foot": depth[y2:min(h, y2 + pad_y), x1:x2]
        }

        accessibility = {}
        stats = {}

        for side, region in regions.items():

            valid = region[region > 0]

            if valid.size < 50:
                accessibility[side] = "unknown"
                stats[side] = {}
                continue

            median_depth = np.median(valid)
            depth_diff = median_depth - bed_depth

            far_pixels = valid > (bed_depth + self.wall_offset)
            far_percent = np.sum(far_pixels) / len(valid)

            # -------------------------
            # Enhanced logic
            # -------------------------
            label = "partially_blocked"  # default

            # Head is almost always blocked
            if side == "head":
                label = "blocked" if far_percent > 0.5 else "partially_blocked"

            # Foot is mostly free
            elif side == "foot":
                label = "free" if far_percent < 0.65 else "partially_blocked"

            # Left / Right: obstacle aware
            elif side in ["left", "right"]:
                label = "partially_blocked" if obstacles and self._region_has_obstacle(region, obstacles) else "free"

            accessibility[side] = label

            stats[side] = {
                "median_depth": float(median_depth),
                "depth_difference": float(depth_diff),
                "far_percent": float(far_percent)
            }

        return accessibility, stats

    # ------------------------------------------------------------
    def _region_has_obstacle(self, region_depth, obstacles):
        """Check if any obstacle overlaps with this region"""
        # Simple placeholder: if obstacles exist, mark as partial
        return len(obstacles) > 0

    # ------------------------------------------------------------
    def visualize_accessibility(self, rgb, depth, bed_bbox,
                                accessibility, stats):

        output = rgb.copy()
        x1, y1, x2, y2 = map(int, bed_bbox)

        color_map = {
            "free": (0, 255, 0),
            "blocked": (0, 0, 255),
            "partially_blocked": (0, 255, 255),
            "unknown": (128, 128, 128)
        }

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        positions = {
            "left": (x1 - 100, (y1 + y2) // 2),
            "right": (x2 + 10, (y1 + y2) // 2),
            "head": ((x1 + x2) // 2, y1 - 20),
            "foot": ((x1 + x2) // 2, y2 + 20)
        }

        for side, label in accessibility.items():
            color = color_map.get(label, (255, 255, 255))
            pos = positions[side]

            cv2.putText(output,
                        f"{side}: {label}",
                        pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        return output
