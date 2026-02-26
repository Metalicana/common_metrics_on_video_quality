import sys
import os
import torch
import numpy as np
import argparse
import cv2
import json
from tqdm import tqdm

# Force stable OpenCV backend and single-threading to prevent engine errors
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"

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
        tii_root = os.path.join(data_root, "TII", "videos")
        uzh_root = os.path.join(data_root, "UZH", "videos")

        with open(id_manifest, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                raw_part = line.split(',')[0].strip()
                vid_name = os.path.splitext(os.path.basename(raw_part))[0]
                if vid_name == "frame0":
                    vid_name = os.path.basename(os.path.dirname(raw_part))

                gen_path = os.path.join(self.generated_dir, f"{vid_name}.mp4")
                gt_path = next((p for p in [os.path.join(tii_root, f"{vid_name}.mp4"), 
                                            os.path.join(uzh_root, f"{vid_name}.mp4")] 
                                if os.path.exists(p)), None)
                
                if os.path.exists(gen_path) and gt_path:
                    self.samples.append({"gen": gen_path, "gt": gt_path, "id": vid_name})

    def load_video(self, path):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        frames = []
        try:
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret or frame is None: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
        finally:
            cap.release()
            
        if not frames: return None
        while len(frames) < self.num_frames: frames.append(frames[-1])
        video_np = np.array(frames[:self.num_frames])
        # Return on CPU initially to avoid early device conflicts
        return torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

def extract_val(res, key_hint):
    if isinstance(res, dict):
        key = next((k for k in res.keys() if key_hint in k.lower()), list(res.keys())[0])
        val = res[key]
    else:
        val = res
    # Ensure value is off GPU before conversion
    if torch.is_tensor(val):
        val = val.detach().cpu().item() if val.numel() == 1 else val.detach().cpu().numpy()
    return np.mean(val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--id_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestVideoDataset(args.generated_dir, args.id_manifest, args.data_root, 
                                  num_frames=args.frames, size=args.size)
    
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    valid_gen_videos = []
    valid_gt_videos = []

    print(f"--- 📉 Calculating PSNR/SSIM/LPIPS for {len(dataset.samples)} videos ---")
    for sample in tqdm(dataset.samples):
        try:
            v_gen_cpu = dataset.load_video(sample['gen'])
            v_gt_cpu = dataset.load_video(sample['gt'])
            
            if v_gen_cpu is None or v_gt_cpu is None: continue

            # Move to device for calculation
            v_gen = v_gen_cpu.unsqueeze(0).to(device)
            v_gt = v_gt_cpu.unsqueeze(0).to(device)

            # Metric calls - ensure these functions handle device internal to their logic
            # or explicitly use .detach().cpu() within extract_val
            metrics["psnr"].append(extract_val(calculate_psnr(v_gen, v_gt), 'psnr'))
            metrics["ssim"].append(extract_val(calculate_ssim(v_gen, v_gt), 'ssim'))
            metrics["lpips"].append(extract_val(calculate_lpips(v_gen, v_gt, device), 'lpips'))
            
            # Save to CPU for FVD to prevent GPU OOM
            valid_gen_videos.append(v_gen_cpu.unsqueeze(0))
            valid_gt_videos.append(v_gt_cpu.unsqueeze(0))
            
            del v_gen, v_gt
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError skipping {sample['id']}: {e}")

    if not metrics["psnr"]:
        print("❌ No videos were processed successfully.")
        return

    # Calculate Global FVD
    print(f"--- 🧠 Calculating Global FVD for {len(valid_gen_videos)} videos ---")
    try:
        # Concatenate on CPU, then calculate_fvd usually handles the GPU transfer internally
        all_gen = torch.cat(valid_gen_videos, dim=0)
        all_gt = torch.cat(valid_gt_videos, dim=0)
        fvd_res = calculate_fvd(all_gen, all_gt, device, method='styleganv')
        fvd_val = extract_val(fvd_res, 'fvd')
    except Exception as e:
        print(f"FVD Calculation failed: {e}")
        fvd_val = float('nan')

    final_results = {
        "psnr": float(np.mean(metrics["psnr"])),
        "ssim": float(np.mean(metrics["ssim"])),
        "lpips": float(np.mean(metrics["lpips"])),
        "fvd": fvd_val,
        "num_samples": len(metrics["psnr"])
    }

    print("\n" + json.dumps(final_results, indent=4))
    
    out_file = os.path.join(args.generated_dir, "visual_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_captions--id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_captions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81