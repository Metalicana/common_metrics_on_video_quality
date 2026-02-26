import sys
import os
import torch
import numpy as np
import argparse
import cv2
import json
from tqdm import tqdm

# Force stable OpenCV backend
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestVideoDataset(args.generated_dir, args.id_manifest, args.data_root, 
                                  num_frames=args.frames, size=args.size)
    
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    all_gen_features = []
    all_gt_features = []

    print(f"--- 📉 Processing {len(dataset.samples)} videos ---")
    for sample in tqdm(dataset.samples):
        try:
            v_gen = dataset.load_video(sample['gen']).unsqueeze(0).to(device)
            v_gt = dataset.load_video(sample['gt']).unsqueeze(0).to(device)

            # Standard Metrics
            metrics["psnr"].append(extract_val(calculate_psnr(v_gen, v_gt), 'psnr'))
            metrics["ssim"].append(extract_val(calculate_ssim(v_gen, v_gt), 'ssim'))
            metrics["lpips"].append(extract_val(calculate_lpips(v_gen, v_gt, device), 'lpips'))
            
            # Feature Extraction for FVD (Assume calculate_fvd can return features)
            # If your calculate_fvd only does end-to-end, we must pass it in chunks
            # but for true FVD, we aggregate.
            all_gen_features.append(v_gen.cpu()) # Still big, but better than batching FVD
            all_gt_features.append(v_gt.cpu())
            
            del v_gen, v_gt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error skipping {sample['id']}: {e}")

    # Calculate Global FVD
    print(f"--- 🧠 Calculating Global FVD ---")
    # This assumes calculate_fvd supports the full stack. 
    # If memory is STILL an issue, calculate_fvd needs to be modified 
    # to return the I3D features instead of the distance.
    gen_tensor = torch.cat(all_gen_features, dim=0)
    gt_tensor = torch.cat(all_gt_features, dim=0)
    
    fvd_res = calculate_fvd(gen_tensor, gt_tensor, device, method='styleganv')
    
    final_results = {
        "psnr_mean": np.mean(metrics["psnr"]),
        "ssim_mean": np.mean(metrics["ssim"]),
        "lpips_mean": np.mean(metrics["lpips"]),
        "fvd": extract_val(fvd_res, 'fvd'),
        "num_samples": len(metrics["psnr"])
    }

    print("\n" + json.dumps(final_results, indent=4))
    with open(os.path.join(args.generated_dir, "visual_metrics.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_captions--id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_captions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81