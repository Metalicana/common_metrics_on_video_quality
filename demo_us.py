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

def extract_val(res, key_hint):
    if isinstance(res, dict):
        key = next((k for k in res.keys() if key_hint in k.lower()), list(res.keys())[0])
        val = res[key]
    else:
        val = res
    return np.mean(val) if isinstance(val, (list, np.ndarray, torch.Tensor)) else float(val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--id_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--fvd_batch", type=int, default=16) # Memory-safe batch size
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestVideoDataset(args.generated_dir, args.id_manifest, args.data_root, num_frames=args.frames, size=args.size)
    
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    all_gen_cpu, all_gt_cpu = [], []
    count = 0

    print(f"--- 📉 Calculating PSNR/SSIM/LPIPS for {len(dataset.samples)} videos ---")
    for sample in tqdm(dataset.samples):
        try:
            v_gen = dataset.load_video(sample['gen']).unsqueeze(0)
            v_gt = dataset.load_video(sample['gt']).unsqueeze(0)

            psnr_total += extract_val(calculate_psnr(v_gen, v_gt), 'psnr')
            ssim_total += extract_val(calculate_ssim(v_gen, v_gt), 'ssim')
            lpips_total += extract_val(calculate_lpips(v_gen, v_gt, device), 'lpips')
            
            # Store on CPU to avoid "Killed" OOM
            all_gen_cpu.append(v_gen.cpu())
            all_gt_cpu.append(v_gt.cpu())
            count += 1
        except Exception as e:
            print(f"Error skipping {sample['id']}: {e}")

    # Final pixel averages
    result = {
        "psnr": psnr_total / count,
        "ssim": ssim_total / count,
        "lpips": lpips_total / count,
        "num_samples": count
    }

    print(f"--- 🧠 Calculating FVD in batches of {args.fvd_batch} ---")
    fvd_accum = 0.0
    fvd_chunks = 0
    
    for i in range(0, len(all_gen_cpu), args.fvd_batch):
        batch_gen = torch.cat(all_gen_cpu[i : i + args.fvd_batch], dim=0)
        batch_gt = torch.cat(all_gt_cpu[i : i + args.fvd_batch], dim=0)
        
        # Batch size must be > 1 for FVD covariance math
        if batch_gen.shape[0] < 2: continue

        res_fvd = calculate_fvd(batch_gen, batch_gt, device, method='styleganv')
        fvd_accum += extract_val(res_fvd, 'fvd')
        fvd_chunks += 1
        
        # Explicitly free GPU memory
        del batch_gen, batch_gt
        torch.cuda.empty_cache()

    result['fvd'] = fvd_accum / fvd_chunks

    print("\n" + json.dumps(result, indent=4))
    
    out_file = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/vanilla_finetune   --id_manifest /home/ab575577/projects_spring_2026/AeroBench/manifests/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81