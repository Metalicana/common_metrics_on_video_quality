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

try:
    from calculate_fvd import calculate_fvd
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim
    from calculate_lpips import calculate_lpips
except ImportError:
    print("❌ Error: Metric scripts not found.")
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

        with open(id_manifest, 'r') as f:
            for line in f:
                if not line.strip(): continue
                raw_part = line.split(',')[0].strip()
                vid_name = os.path.splitext(os.path.basename(raw_part))[0]
                if vid_name == "frame0":
                    vid_name = os.path.basename(os.path.dirname(raw_part))

                gen_path = os.path.join(self.generated_dir, f"{vid_name}.mp4")
                gt_path = None
                for root in [tii_video_root, uzh_video_root]:
                    p = os.path.join(root, f"{vid_name}.mp4")
                    if os.path.exists(p):
                        gt_path = p
                        break
                
                if os.path.exists(gen_path) and gt_path:
                    self.samples.append({"gen": gen_path, "gt": gt_path, "id": vid_name})

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        if not frames: return None
        while len(frames) < self.num_frames: frames.append(frames[-1])
        video_np = np.array(frames[:self.num_frames])
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
    
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    all_gen_for_fvd, all_gt_for_fvd = [], []
    count = 0

    print(f"--- 📉 Calculating Metrics in loop for {len(dataset.samples)} videos ---")
    for sample in tqdm(dataset.samples):
        v_gen = dataset.load_video(sample['gen']).unsqueeze(0) # [1, T, C, H, W]
        v_gt = dataset.load_video(sample['gt']).unsqueeze(0)

        # 1. Pixel/Perceptual metrics (Immediate cleanup)
        psnr_total += calculate_psnr(v_gen, v_gt)
        ssim_total += calculate_ssim(v_gen, v_gt)
        lpips_total += calculate_lpips(v_gen, v_gt, device)
        
        # 2. Collect for FVD (We keep these, but they are fewer if FVD implementation allows)
        # Note: FVD usually needs at least 16 videos. We collect all but move to CPU to save GPU RAM
        all_gen_for_fvd.append(v_gen.cpu())
        all_gt_for_fvd.append(v_gt.cpu())
        
        count += 1

    # Final averages
    result = {
        "psnr": psnr_total / count,
        "ssim": ssim_total / count,
        "lpips": lpips_total / count,
    }

    print("--- 🧠 Calculating FVD (Aggregated) ---")
    # FVD still needs the stack, but we do it last
    videos_gen = torch.cat(all_gen_for_fvd, dim=0)
    videos_gt = torch.cat(all_gt_for_fvd, dim=0)
    
    # StyleGAN-V FVD
    result['fvd'] = calculate_fvd(videos_gen, videos_gt, device, method='styleganv')

    print("\n" + json.dumps(result, indent=4))
    
    out_file = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()