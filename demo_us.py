import sys
import os
import torch
import numpy as np
import argparse
import cv2
import json
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.getcwd())

# Ensure internal metrics can be imported
try:
    from calculate_fvd import calculate_fvd
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim
    from calculate_lpips import calculate_lpips
except ImportError:
    print("❌ Error: Metric scripts not found. Run from the directory containing calculate_fvd.py")
    sys.exit(1)

class ManifestVideoDataset:
    def __init__(self, generated_dir, id_manifest, data_root, num_frames=81, size=384):
        self.generated_dir = os.path.abspath(generated_dir)
        self.num_frames = num_frames
        self.size = size
        self.samples = []

        data_root = os.path.abspath(data_root)
        tii_video_root = os.path.join(data_root, "TII", "videos")
        uzh_video_root = os.path.join(data_root, "UZH", "videos")

        print(f"--- 📖 Reading IDs from: {id_manifest} ---")
        with open(id_manifest, 'r') as f:
            for line in f:
                if not line.strip(): continue
                
                raw_part = line.split(',')[0].strip()
                vid_name = os.path.splitext(os.path.basename(raw_part))[0]
                
                if vid_name == "frame0":
                    vid_name = os.path.basename(os.path.dirname(raw_part))

                gen_path = os.path.join(self.generated_dir, f"{vid_name}.mp4")
                
                # Check both TII and UZH for ground truth
                gt_path = None
                for root in [tii_video_root, uzh_video_root]:
                    p = os.path.join(root, f"{vid_name}.mp4")
                    if os.path.exists(p):
                        gt_path = p
                        break
                
                if os.path.exists(gen_path) and gt_path:
                    self.samples.append({"gen": gen_path, "gt": gt_path, "id": vid_name})

        print(f"--- ✅ Found {len(self.samples)} aligned pairs ---")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            # INTER_AREA is more rigorous for downsampling to 384x384
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        
        if not frames: return None
        while len(frames) < self.num_frames: frames.append(frames[-1])
        
        video_np = np.array(frames[:self.num_frames])
        # Return [T, C, H, W] normalized to [0, 1]
        return torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

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
        print("❌ No aligned videos found.")
        return

    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    all_gen_for_fvd, all_gt_for_fvd = [], []
    count = 0

    print(f"--- 📉 Calculating Metrics iteratively for {len(dataset.samples)} videos ---")
    for sample in tqdm(dataset.samples):
        v_gen = dataset.load_video(sample['gen']).unsqueeze(0) # [1, T, C, H, W]
        v_gt = dataset.load_video(sample['gt']).unsqueeze(0)

        # Extract values from dicts returned by calculation scripts
        res_psnr = calculate_psnr(v_gen, v_gt)
        psnr_total += res_psnr['psnr'] if isinstance(res_psnr, dict) else res_psnr
        
        res_ssim = calculate_ssim(v_gen, v_gt)
        ssim_total += res_ssim['ssim'] if isinstance(res_ssim, dict) else res_ssim
        
        res_lpips = calculate_lpips(v_gen, v_gt, device)
        lpips_total += res_lpips['lpips'] if isinstance(res_lpips, dict) else res_lpips
        
        # Keep on CPU to save System RAM; move to GPU only during FVD call
        all_gen_for_fvd.append(v_gen.cpu())
        all_gt_for_fvd.append(v_gt.cpu())
        
        count += 1

    # Aggregated averages
    final_psnr = psnr_total / count
    final_ssim = ssim_total / count
    final_lpips = lpips_total / count

    print(f"PSNR: {final_psnr:.4f} | SSIM: {final_ssim:.4f} | LPIPS: {final_lpips:.4f}")

    print("--- 🧠 Aggregating Tensors for FVD ---")
    # This part consumes significant RAM; Cat on CPU first
    videos_gen = torch.cat(all_gen_for_fvd, dim=0)
    videos_gt = torch.cat(all_gt_for_fvd, dim=0)

    # StyleGAN-V FVD
    res_fvd = calculate_fvd(videos_gen, videos_gt, device, method='styleganv')
    final_fvd = res_fvd['fvd'] if isinstance(res_fvd, dict) else res_fvd

    result = {
        "psnr": final_psnr,
        "ssim": final_ssim,
        "lpips": final_lpips,
        "fvd": final_fvd,
        "num_samples": count
    }

    print("\n" + json.dumps(result, indent=4))
    
    out_file = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"💾 Results saved to {out_file}")

if __name__ == "__main__":
    main()