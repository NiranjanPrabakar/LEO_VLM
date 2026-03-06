from ultralytics import YOLO
from accessibility_analyzer import AccessibilityAnalyzer
from class_config import CLASS_NAMES, get_class_category
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def process_dataset():
    """Process all keyframes using enhanced depth + obstacle logic"""

    # -------------------------------------------------------------
    # Load YOLO model
    # -------------------------------------------------------------
    model = YOLO('models/bed_detector_v1/weights/best.pt')

    # -------------------------------------------------------------
    # Initialize analyzer
    # -------------------------------------------------------------
    analyzer = AccessibilityAnalyzer(
        wall_threshold_offset=500,
        free_threshold_offset=300,
        blocked_percent_threshold=0.65
    )

    keyframes_dir = Path('data/keyframes')
    output_file = 'results/accessibility_dataset.json'
    viz_dir = Path('results/accessibility_visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)

    dataset = []

    rgb_files = sorted(keyframes_dir.glob('*_rgb.png'))

    print("\n" + "="*60)
    print(f"Processing {len(rgb_files)} frames with enhanced obstacle + depth logic...")
    print("="*60 + "\n")

    for rgb_path in tqdm(rgb_files, desc="Analyzing frames"):

        frame_id = rgb_path.stem.replace('_rgb', '')
        depth_path = keyframes_dir / f'{frame_id}_depth.npy'

        if not depth_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        depth = np.load(str(depth_path))

        # Run YOLO
        results = model(rgb, conf=0.25, verbose=False)

        detections = {'bed': None, 'bedding': [], 'pillows': [], 'obstacles': []}

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            category = get_class_category(class_name)

            bbox = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0])

            detection_info = {'class': class_name, 'bbox': bbox, 'confidence': confidence}

            if category == 'bed':
                if detections['bed'] is None or confidence > detections['bed']['confidence']:
                    detections['bed'] = detection_info
            elif category == 'bedding':
                detections['bedding'].append(detection_info)
            elif category == 'pillow':
                detections['pillows'].append(detection_info)
            elif category == 'obstacle':
                detections['obstacles'].append(detection_info)

        # Accessibility analysis
        accessibility, accessibility_stats = None, None
        if detections['bed'] is not None:
            bed_bbox = detections['bed']['bbox']

            obstacles = [obs['bbox'] for obs in detections['obstacles']] if detections['obstacles'] else []
            accessibility, accessibility_stats = analyzer.analyze_bed_accessibility(
                depth, bed_bbox, obstacles
            )

            if accessibility is not None:
                viz = analyzer.visualize_accessibility(
                    rgb, depth, bed_bbox, accessibility, accessibility_stats
                )
                viz_path = viz_dir / f'{frame_id}_accessibility.png'
                cv2.imwrite(str(viz_path), viz)

        # Store frame results
        frame_data = {
            'frame_id': frame_id,
            'detections': detections,
            'accessibility': accessibility,
            'depth_stats': accessibility_stats
        }

        dataset.append(frame_data)

    # Save dataset
    Path('results').mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)

    beds_detected = sum(1 for d in dataset if d['detections']['bed'] is not None)
    print(f"Total frames: {len(dataset)}")
    print(f"Beds detected: {beds_detected}")

    if beds_detected > 0:
        print("\nAccessibility Summary:")
        for side in ['left', 'right', 'head', 'foot']:
            free = sum(1 for d in dataset if d['accessibility'] and d['accessibility'].get(side) == 'free')
            blocked = sum(1 for d in dataset if d['accessibility'] and d['accessibility'].get(side) == 'blocked')
            partial = sum(1 for d in dataset if d['accessibility'] and d['accessibility'].get(side) == 'partially_blocked')
            print(f"{side.capitalize():5s}: {free:4d} free, {partial:4d} partial, {blocked:4d} blocked")

    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ Visualizations saved to: {viz_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    process_dataset()
