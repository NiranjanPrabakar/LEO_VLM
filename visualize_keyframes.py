import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - ADD THIS FIRST!

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

def visualize_rgbd_pair(rgb_path, depth_path, save_path=None):
    """Visualize RGB and depth side-by-side"""
    
    # Load RGB
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load Depth
    depth = np.load(str(depth_path))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB Image\n{rgb_path.stem}')
    axes[0].axis('off')
    
    # Handle depth visualization
    depth_viz = depth.copy()
    # Filter out zero/invalid values for better visualization
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        vmin, vmax = np.percentile(valid_depth, [5, 95])
    else:
        vmin, vmax = 0, 1
    
    im = axes[1].imshow(depth_viz, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('Depth Map (mm)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Depth (mm)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.close()

def visualize_random_samples(num_samples=10):
    """Visualize random samples from extracted keyframes"""
    
    keyframes_dir = Path('data/keyframes')
    
    # Get all RGB files
    rgb_files = list(keyframes_dir.glob('*_rgb.png'))
    
    if len(rgb_files) == 0:
        print("No keyframes found! Run extract_keyframes.py first.")
        return
    
    print(f"Found {len(rgb_files)} RGB frames")
    
    # Sample random frames
    samples = random.sample(rgb_files, min(num_samples, len(rgb_files)))
    
    # Create visualization directory
    viz_dir = Path('results/visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(samples)} visualizations...")
    
    for i, rgb_path in enumerate(samples):
        # Get corresponding depth file
        depth_path = rgb_path.parent / rgb_path.name.replace('_rgb.png', '_depth.npy')
        
        if not depth_path.exists():
            print(f"Warning: Depth file not found for {rgb_path.name}")
            continue
        
        # Save visualization
        save_path = viz_dir / f'visualization_{i+1:02d}_{rgb_path.stem}.png'
        visualize_rgbd_pair(rgb_path, depth_path, save_path)
    
    print(f"\n✓ Visualizations saved to: {viz_dir.absolute()}")

def generate_dataset_summary():
    """Generate summary statistics of extracted dataset"""
    
    keyframes_dir = Path('data/keyframes')
    log_path = keyframes_dir / 'extraction_log.json'
    
    if not log_path.exists():
        print("Extraction log not found!")
        return
    
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    # Analyze distribution
    date_counts = {}
    video_counts = {}
    
    for entry in log:
        date = entry['date_folder']
        video = entry['video_num']
        
        date_counts[date] = date_counts.get(date, 0) + 1
        video_counts[f"{date}/{video}"] = video_counts.get(f"{date}/{video}", 0) + 1
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames extracted: {len(log)}")
    print(f"\nFrames per date:")
    for date in sorted(date_counts.keys()):
        print(f"  {date}: {date_counts[date]} frames")
    
    print(f"\nTop 10 videos by frame count:")
    sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)
    for video, count in sorted_videos[:10]:
        print(f"  {video}: {count} frames")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Generate visualizations for 10 random samples
    visualize_random_samples(num_samples=10)
    
    # Print dataset summary
    generate_dataset_summary()
