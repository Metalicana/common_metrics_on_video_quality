import sys
import os

# Add the current directory to sys.path to resolve internal metric imports
sys.path.append(os.getcwd())

import torch
import numpy as np
import argparse
import yaml
import cv2
import json
from tqdm import tqdm

# Ensure these files are in your python path
try:
    from calculate_fvd import calculate_fvd
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim
    from calculate_lpips import calculate_lpips
except ImportError:
    print("❌ Error: Metric calculation scripts (calculate_fvd.py, etc.) not found in path.")
    sys.exit(1)

class ManifestVideoDataset:
    def __init__(self, generated_dir, id_manifest, data_root, num_frames=81, size=384):
        self.generated_dir = generated_dir
        self.num_frames = num_frames
        self.size = size
        self.samples = []

        # Data source mapping
        search_paths = {
            "TII": os.path.join(data_root, "TII", "videos"),
            "UZH": os.path.join(data_root, "UZH", "videos")
        }

        print(f"--- 📖 Reading IDs from: {id_manifest} ---")
        with open(id_manifest, 'r') as f:
            ids = [l.strip().split(',')[0].strip() for l in f if l.strip()]

        for vid_id in ids:
            # 1. Generated video path
            gen_path = os.path.join(generated_dir, f"{vid_id}.mp4")
            if not os.path.exists(gen_path):
                continue

            # 2. GT video path (Look in TII then UZH)
            gt_path = None
            for source in ["TII", "UZH"]:
                v_path = os.path.join(search_paths[source], f"{vid_id}.mp4")
                if os.path.exists(v_path):
                    gt_path = v_path
                    break
            
            if gt_path:
                self.samples.append({"gen": gen_path, "gt": gt_path, "id": vid_id})

        print(f"--- ✅ Found {len(self.samples)} aligned pairs for visual metrics ---")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (self.size, self.size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if not frames: return None
        while len(frames) < self.num_frames: frames.append(frames[-1])
        
        # [T, H, W, C] -> [T, C, H, W]
        video_np = np.array(frames[:self.num_frames])
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
        return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True, help="Path to generated mp4s")
    parser.add_argument("--id_manifest", required=True, help="val_subset.txt")
    parser.add_argument("--data_root", required=True, help="AeroBench root folder")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--only_final", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Dataset
    dataset = ManifestVideoDataset(
        args.generated_dir, args.id_manifest, args.data_root, 
        num_frames=args.frames, size=args.size
    )
    
    if len(dataset.samples) == 0:
        print("❌ No videos found to evaluate.")
        return

    # 2. Load all videos into memory (Warning: High RAM usage for H100 nodes)
    # If RAM is an issue, we can move this inside a loop
    gen_list = []
    gt_list = []
    
    print("--- 📂 Loading Videos into Tensors ---")
    for sample in tqdm(dataset.samples):
        gen_v = dataset.load_video(sample['gen'])
        gt_v = dataset.load_video(sample['gt'])
        if gen_v is not None and gt_v is not None:
            gen_list.append(gen_v)
            gt_list.append(gt_v)

    # Stack to [N, T, C, H, W]
    videos_gen = torch.stack(gen_list)
    videos_gt = torch.stack(gt_list)

    # 3. Run Metrics
    print(f"--- 📉 Calculating Metrics on {len(gen_list)} videos ---")
    result = {}
    
    # Standard formats: PSNR/SSIM usually [N, T, C, H, W]
    result['psnr'] = calculate_psnr(videos_gen, videos_gt, only_final=args.only_final)
    result['ssim'] = calculate_ssim(videos_gen, videos_gt, only_final=args.only_final)
    
    # LPIPS/FVD often need GPU
    result['lpips'] = calculate_lpips(videos_gen, videos_gt, device, only_final=args.only_final)
    
    # FVD check: many implementations require [N, C, T, H, W]
    # Check your calculate_fvd.py; if it fails, try permuting to (0, 2, 1, 3, 4)
    result['fvd'] = calculate_fvd(videos_gen, videos_gt, device, method='styleganv', only_final=args.only_final)

    # 4. Output
    print("\n--- 🏆 Final Results ---")
    print(json.dumps(result, indent=4))
    
    output_path = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"💾 Saved results to {output_path}")

if __name__ == "__main__":
    main()