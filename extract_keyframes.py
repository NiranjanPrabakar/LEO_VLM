import cv2
import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import json

class KeyframeExtractor:
    def __init__(self, base_path='/home/hp/Documents', camera_id='cam4'):
        """
        base_path: Path to Documents folder containing date folders
        camera_id: 'cam1', 'cam2', 'cam3', or 'cam4'
        """
        self.base_path = Path(base_path)
        self.camera_id = camera_id
        self.output_dir = Path('data/keyframes')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self.extraction_log = []
        
    def get_all_recordings(self):
        """
        Find all recording folders across all dates
        Returns: List of (date_folder, video_number) tuples
        """
        date_folders = [
            '02 FEB 2026 3',
            '02 FEB 2026 4', 
            '03 FEB 2026 1',
            '03 FEB 2026 2',
            '03 FEB 2026 3',
            '03 FEB 2026 4',
            '04 Feb 2026 1',
            '04 Feb 2026 2',
            '05 FEB 2026 1',
            '05 FEB 2026 2',
            '05 FEB 2026 3',
            '05 FEB 2026 4'
        ]
        
        recordings = []
        for date_folder in date_folders:
            date_path = self.base_path / date_folder
            if not date_path.exists():
                print(f"Warning: {date_folder} not found, skipping...")
                continue
                
            # Get all numbered subfolders
            subfolders = [f for f in date_path.iterdir() if f.is_dir() and f.name.isdigit()]
            subfolders.sort(key=lambda x: int(x.name))
            
            for subfolder in subfolders:
                video_num = int(subfolder.name)
                if 1 <= video_num <= 41:  # Only process videos 1-41
                    recordings.append((date_folder, video_num))
        
        return recordings
    
    def load_timestamps(self, recording_path):
        """Load RGB and depth timestamps for synchronization"""
        try:
            rgb_csv = recording_path / 'timestamps' / f'{self.camera_id}_d435i_{self.camera_id}_color_image_raw.csv'
            depth_csv = recording_path / 'timestamps' / f'{self.camera_id}_d435i_{self.camera_id}_aligned_depth_to_color_image_raw.csv'
            
            if not rgb_csv.exists() or not depth_csv.exists():
                return None, None
            
            rgb_timestamps = pd.read_csv(rgb_csv)
            depth_timestamps = pd.read_csv(depth_csv)
            
            return rgb_timestamps, depth_timestamps
        except Exception as e:
            print(f"Error loading timestamps: {e}")
            return None, None
    
    def extract_frames_from_recording(self, date_folder, video_num, interval_seconds=5, max_frames_per_video=10):
        """
        Extract keyframes from a single recording
        
        interval_seconds: Extract one frame every N seconds
        max_frames_per_video: Maximum number of frames to extract per video
        """
        # Build recording path
        recording_path = self.base_path / date_folder / str(video_num)
        
        if not recording_path.exists():
            return 0
        
        # Check if required files exist
        video_path = recording_path / 'videos' / f'{self.camera_id}_d435i_{self.camera_id}_color_image_raw.mp4'
        depth_path = recording_path / 'depth_frames' / f'{self.camera_id}_d435i_{self.camera_id}_aligned_depth_to_color_image_raw.h5'
        
        if not video_path.exists():
            return 0
        
        if not depth_path.exists():
            return 0
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            cap.release()
            return 0
        
        frame_interval = int(fps * interval_seconds)
        
        # Load depth data
        try:
            depth_file = h5py.File(depth_path, 'r')
            depth_dataset = depth_file['data']
        except Exception as e:
            print(f"Error opening depth file {depth_path}: {e}")
            cap.release()
            return 0
        
        # Load timestamps
        rgb_timestamps, depth_timestamps = self.load_timestamps(recording_path)
        
        if rgb_timestamps is None or depth_timestamps is None:
            cap.release()
            depth_file.close()
            return 0
        
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames_per_video:
            ret, rgb_frame = cap.read()
            if not ret:
                break
            
            # Extract at intervals
            if frame_count % frame_interval == 0:
                # Find matching depth frame
                if frame_count < len(rgb_timestamps):
                    rgb_time = rgb_timestamps.iloc[frame_count]['ros_time_s']  # FIXED: Changed from 'timestamp'
                    
                    # Find closest depth timestamp
                    depth_idx = (depth_timestamps['ros_time_s'] - rgb_time).abs().argmin()  # FIXED: Changed from 'timestamp'
                    
                    # Load depth frame
                    if depth_idx < len(depth_dataset):
                        depth_frame = depth_dataset[depth_idx]
                        
                        # Create unique filename with date and video info
                        date_code = date_folder.replace(' ', '_').replace('FEB', 'Feb')
                        filename_prefix = f'{date_code}_vid{video_num:02d}_frame{extracted_count:03d}'
                        
                        # Save RGB
                        rgb_filename = f'{filename_prefix}_rgb.png'
                        cv2.imwrite(str(self.output_dir / rgb_filename), rgb_frame)
                        
                        # Save Depth
                        depth_filename = f'{filename_prefix}_depth.npy'
                        np.save(str(self.output_dir / depth_filename), depth_frame)
                        
                        # Log extraction
                        self.extraction_log.append({
                            'date_folder': date_folder,
                            'video_num': video_num,
                            'frame_count': frame_count,
                            'rgb_filename': rgb_filename,
                            'depth_filename': depth_filename,
                            'timestamp': float(rgb_time)
                        })
                        
                        extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        depth_file.close()
        
        return extracted_count
    
    def extract_all(self, interval_seconds=5, max_frames_per_video=10):
        """
        Extract keyframes from ALL recordings across all dates
        
        interval_seconds: Extract one frame every N seconds
        max_frames_per_video: Max frames per video
        """
        recordings = self.get_all_recordings()
        
        print(f"\n{'='*60}")
        print(f"Found {len(recordings)} recordings to process")
        print(f"Camera: {self.camera_id}")
        print(f"Interval: {interval_seconds}s")
        print(f"Max frames per video: {max_frames_per_video}")
        print(f"{'='*60}\n")
        
        total_extracted = 0
        successful_recordings = 0
        failed_recordings = []
        
        for date_folder, video_num in tqdm(recordings, desc="Processing recordings"):
            try:
                count = self.extract_frames_from_recording(
                    date_folder, 
                    video_num, 
                    interval_seconds=interval_seconds,
                    max_frames_per_video=max_frames_per_video
                )
                
                if count > 0:
                    total_extracted += count
                    successful_recordings += 1
                else:
                    failed_recordings.append((date_folder, video_num))
                    
            except Exception as e:
                print(f"\nError processing {date_folder}/{video_num}: {e}")
                failed_recordings.append((date_folder, video_num))
        
        # Save extraction log
        log_path = self.output_dir / 'extraction_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.extraction_log, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total recordings processed: {len(recordings)}")
        print(f"Successful: {successful_recordings}")
        print(f"Failed: {len(failed_recordings)}")
        print(f"Total frames extracted: {total_extracted}")
        print(f"Average per successful recording: {total_extracted/successful_recordings if successful_recordings > 0 else 0:.1f}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Extraction log: {log_path.absolute()}")
        
        if failed_recordings and len(failed_recordings) <= 20:
            print(f"\nFailed recordings:")
            for date, vid in failed_recordings:
                print(f"  - {date}/{vid}")
        elif failed_recordings:
            print(f"\nFailed recordings (showing first 10):")
            for date, vid in failed_recordings[:10]:
                print(f"  - {date}/{vid}")
            print(f"  ... and {len(failed_recordings)-10} more")
        
        print(f"{'='*60}\n")
        
        return total_extracted

# Main execution
if __name__ == "__main__":
    # Initialize extractor
    extractor = KeyframeExtractor(
        base_path='/home/hp/Documents',
        camera_id='cam4'
    )
    
    # Extract frames from all recordings
    total = extractor.extract_all(
        interval_seconds=5,
        max_frames_per_video=10
    )
    
    print(f"\n✓ Successfully extracted {total} RGB-Depth pairs!")
