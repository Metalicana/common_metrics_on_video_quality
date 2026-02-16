import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import torch
import numpy as np
import argparse
import cv2
import json
from tqdm import tqdm

# Ensure internal metrics can be imported
try:
    from calculate_fvd import calculate_fvd
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim
    from calculate_lpips import calculate_lpips
except ImportError:
    print("❌ Error: Metric scripts not found. Run from the directory containing calculate_fvd.py etc.")
    sys.exit(1)

class ManifestVideoDataset:
    def __init__(self, generated_dir, id_manifest, data_root, num_frames=81, size=384):
        self.generated_dir = os.path.abspath(generated_dir)
        self.num_frames = num_frames
        self.size = size
        self.samples = []

        data_root = os.path.abspath(data_root)
        
        # Exact paths based on your context
        tii_video_root = os.path.join(data_root, "TII", "videos")
        uzh_video_root = os.path.join(data_root, "UZH", "videos")

        print(f"--- 📖 Reading IDs from: {id_manifest} ---")
        with open(id_manifest, 'r') as f:
            for line in f:
                if not line.strip(): continue
                
                # Extract the ID (the name)
                # Handles lines like "/path/to/TII_flight_001.jpg, ..." or just "TII_flight_001"
                raw_part = line.split(',')[0].strip()
                vid_name = os.path.splitext(os.path.basename(raw_part))[0]
                
                # Check if it's "frame0", if so, take parent folder name
                if vid_name == "frame0":
                    vid_name = os.path.basename(os.path.dirname(raw_part))

                # 1. Path to your MODEL GENERATED video
                gen_path = os.path.join(self.generated_dir, f"{vid_name}.mp4")
                
                # 2. Path to GROUND TRUTH
                gt_path = None
                tii_path = os.path.join(tii_video_root, f"{vid_name}.mp4")
                uzh_path = os.path.join(uzh_video_root, f"{vid_name}.mp4")

                if os.path.exists(tii_path):
                    gt_path = tii_path
                elif os.path.exists(uzh_path):
                    gt_path = uzh_path
                
                if os.path.exists(gen_path) and gt_path:
                    self.samples.append({"gen": gen_path, "gt": gt_path, "id": vid_name})

        print(f"--- ✅ Found {len(self.samples)} aligned pairs ---")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            # INTER_AREA is more rigorous for downsampling
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_AREA)
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
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--id_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestVideoDataset(args.generated_dir, args.id_manifest, args.data_root, num_frames=args.frames, size=args.size)
    
    if not dataset.samples:
        print("❌ No videos found. Ensure the filenames in generated_dir match the names in val_subset.txt.")
        return

    gen_list, gt_list = [], []
    print("--- 📂 Loading Videos ---")
    for sample in tqdm(dataset.samples):
        v_gen = dataset.load_video(sample['gen'])
        v_gt = dataset.load_video(sample['gt'])
        if v_gen is not None and v_gt is not None:
            gen_list.append(v_gen)
            gt_list.append(v_gt)

    # Convert to Tensors [N, T, C, H, W]
    videos_gen = torch.stack(gen_list)
    videos_gt = torch.stack(gt_list)

    print(f"--- 📉 Calculating Metrics on {len(gen_list)} videos ---")
    
    # Batch processing is often safer for LPIPS/FVD on high-res
    psnr = calculate_psnr(videos_gen, videos_gt)
    ssim = calculate_ssim(videos_gen, videos_gt)
    lpips_val = calculate_lpips(videos_gen, videos_gt, device)
    fvd_val = calculate_fvd(videos_gen, videos_gt, device, method='styleganv')

    result = {"psnr": psnr, "ssim": ssim, "lpips": lpips_val, "fvd": fvd_val}
    print("\n" + json.dumps(result, indent=4))
    
    out_file = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()