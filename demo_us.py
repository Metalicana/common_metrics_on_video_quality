import sys
import os
import torch
import numpy as np
import argparse
import cv2
import json
from tqdm import tqdm

# Force OpenCV stability
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
                if vid_name == "frame0": vid_name = os.path.basename(os.path.dirname(raw_part))
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
        finally: cap.release()
        if not frames: return None
        while len(frames) < self.num_frames: frames.append(frames[-1])
        return torch.from_numpy(np.array(frames[:self.num_frames])).permute(0, 3, 1, 2).float() / 255.0

def safe_extract(res):
    """Rigorous extraction of scalar from any format."""
    if isinstance(res, dict):
        val = list(res.values())[0]
    else:
        val = res
    if torch.is_tensor(val):
        return val.detach().cpu().float().mean().item()
    return float(np.mean(val))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--id_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestVideoDataset(args.generated_dir, args.id_manifest, args.data_root, args.frames, args.size)
    
    results_list = {"psnr": [], "ssim": [], "lpips": []}
    gen_list_for_fvd, gt_list_for_fvd = [], []

    print(f"--- 📉 Processing {len(dataset.samples)} samples ---")
    for sample in tqdm(dataset.samples):
        try:
            # 1. LOAD TO CPU
            v_gen_cpu = dataset.load_video(sample['gen']).unsqueeze(0)
            v_gt_cpu = dataset.load_video(sample['gt']).unsqueeze(0)

            # 2. CALL PSNR/SSIM ON CPU (Bypass the CUDA-to-Numpy crash)
            # results_list["psnr"].append(safe_extract(calculate_psnr(v_gen_cpu, v_gt_cpu)))
            # results_list["ssim"].append(safe_extract(calculate_ssim(v_gen_cpu, v_gt_cpu)))

            # 3. CALL LPIPS ON GPU (Required for speed/model)
            v_gen_gpu = v_gen_cpu.to(device)
            v_gt_gpu = v_gt_cpu.to(device)
            # results_list["lpips"].append(safe_extract(calculate_lpips(v_gen_gpu, v_gt_gpu, device)))

            # 4. STORE FOR FVD (CPU only to avoid OOM)
            gen_list_for_fvd.append(v_gen_cpu)
            gt_list_for_fvd.append(v_gt_cpu)

            del v_gen_gpu, v_gt_gpu
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed {sample['id']}: {e}")

    # 5. FINAL AGGREGATION
    final = {
        "psnr": np.mean(results_list["psnr"]),
        "ssim": np.mean(results_list["ssim"]),
        "lpips": np.mean(results_list["lpips"]),
        "fvd": 0.0,
        "count": len(results_list["psnr"])
    }

    print("--- 🧠 Final FVD Calculation ---")
    try:
        fvd_res = calculate_fvd(torch.cat(gen_list_for_fvd, dim=0), 
                                torch.cat(gt_list_for_fvd, dim=0), 
                                device, method='styleganv')
        final["fvd"] = safe_extract(fvd_res)
    except Exception as e:
        print(f"FVD Error: {e}")

    print(json.dumps(final, indent=4))
    with open(os.path.join(args.generated_dir, "visual_metrics.json"), "w") as f:
        json.dump(final, f, indent=4)

if __name__ == "__main__":
    main()
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_captions--id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_captions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81