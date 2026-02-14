import torch
import os
import cv2
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import your metric functions
# Ensure these files are in the same directory or python path
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

class VideoLoader:
    def __init__(self, size=224, num_frames=81):
        self.size = size
        self.num_frames = num_frames

    def load_folder(self, folder_path):
        """
        Loads all video files from a folder into a single Tensor.
        Returns: Tensor of shape [N, T, C, H, W] in range [0, 1]
        """
        files = sorted([
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])
        
        if len(files) == 0:
            raise ValueError(f"No videos found in {folder_path}")
            
        print(f"Loading {len(files)} videos from {folder_path}...")
        
        batch_videos = []
        
        for file_path in tqdm(files):
            cap = cv2.VideoCapture(file_path)
            frames = []
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize and BGR -> RGB
                frame = cv2.resize(frame, (self.size, self.size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            # Handle edge cases
            if len(frames) == 0:
                print(f"Warning: Empty video {file_path}")
                continue
                
            # Pad or Truncate
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
            frames = frames[:self.num_frames]
            
            # Convert to numpy [T, H, W, C]
            video_np = np.array(frames)
            batch_videos.append(video_np)
            
        # Stack into [N, T, H, W, C]
        batch_tensor = torch.from_numpy(np.stack(batch_videos))
        
        # Permute to [N, T, C, H, W] for standard metrics
        # (Check your calculate_fvd documentation if it needs [N, C, T, H, W])
        batch_tensor = batch_tensor.permute(0, 1, 4, 2, 3)
        
        # Normalize to [0, 1]
        batch_tensor = batch_tensor.float() / 255.0
        
        return batch_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, help="Path to Generated videos")
    parser.add_argument("--val", required=True, help="Path to Validation/Ground Truth videos")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to use (must match model expectations)")
    parser.add_argument("--size", type=int, default=64, help="Resolution to resize to")
    parser.add_argument("--only_final", action="store_true", help="Calculate metrics only on the final frame")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Videos
    loader = VideoLoader(size=args.size, num_frames=args.frames)
    
    print("--- 📂 Loading Datasets ---")
    videos1 = loader.load_folder(args.gen)   # Generated
    videos2 = loader.load_folder(args.val)   # Ground Truth
    
    # Ensure count matches (Truncate to minimum common denominator)
    min_len = min(len(videos1), len(videos2))
    videos1 = videos1[:min_len]
    videos2 = videos2[:min_len]
    
    print(f"Dataset shapes: {videos1.shape} vs {videos2.shape}")
    print(f"Range: [{videos1.min():.2f}, {videos1.max():.2f}]")

    # 2. Run Metrics
    print("--- 📉 Calculating Metrics ---")
    result = {}
    
    # FVD
    # Note: FVD usually expects [B, T, C, H, W] or [B, C, T, H, W]. 
    # If your lib expects [B, C, T, H, W], uncomment the permute below:
    # videos1_fvd = videos1.permute(0, 2, 1, 3, 4)
    # videos2_fvd = videos2.permute(0, 2, 1, 3, 4)
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=args.only_final)
    
    # SSIM & PSNR (Usually run on CPU or GPU batch-wise)
    result['ssim'] = calculate_ssim(videos1, videos2, only_final=args.only_final)
    result['psnr'] = calculate_psnr(videos1, videos2, only_final=args.only_final)
    
    # LPIPS
    result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=args.only_final)

    # 3. Output
    print("\n--- 🏆 Final Results ---")
    print(json.dumps(result, indent=4))
    
    # Save to file
    with open("metrics_results.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
    
    
# Example usage:
# python run_metrics.py --gen /path/to/my_generation_folder --val /path/to/validation_folder --frames 81 --size 224

# /home/ab575577/projects_spring_2026/AeroBench/TII/videos

# /home/ab575577/projects_fall_2025/Aero-predict/validation_res