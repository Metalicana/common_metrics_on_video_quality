import sys
import os
from unittest import result
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

    all_gen_videos = torch.zeros(250, 81, 3, 384, 384, requires_grad=False)
    all_gt_videos = torch.zeros(250, 81, 3, 384, 384, requires_grad=False)
    # import pdb; pdb.set_trace()
    print(f"--- 📉 Processing {len(dataset.samples)} samples ---")
    i = 0
    for sample in tqdm(dataset.samples):
        try:
            # 1. Load Video
            v_gen_cpu = dataset.load_video(sample['gen']).unsqueeze(0)
            v_gt_cpu = dataset.load_video(sample['gt']).unsqueeze(0)

            all_gen_videos[i] = v_gen_cpu.squeeze(0)
            all_gt_videos[i] = v_gt_cpu.squeeze(0) 
            i+=1
            # 4. CRITICAL: Free GPU memory
            del v_gen_cpu, v_gt_cpu
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed {sample['id']}: {e}")

    # 5. Final Calculation
    print(all_gen_videos.shape, all_gt_videos.shape)
    print(f"--- 🧠 Finalizing FVD Distribution Math ---")
    # gen_feats = torch.cat(all_gen_feats, dim=0)
    # gt_feats = torch.cat(all_gt_feats, dim=0)
    result = {}
    result['fvd'] = calculate_fvd(all_gen_videos, all_gt_videos, device, only_final=False)
    # This call now passes features to the distance function
    result['ssim'] = calculate_ssim(all_gen_videos, all_gt_videos, only_final=False)
    result['psnr'] = calculate_psnr(all_gen_videos, all_gt_videos, only_final=False)
    # result['lpips'] = calculate_lpips(all_gen_videos, all_gt_videos, device, only_final=False)
    
    
    
    print(result)

    print(json.dumps(result, indent=4))
    with open(os.path.join(args.generated_dir, f"visual_metrics_{args.generated_dir.split('/')[-1]}.json"), "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_captions--id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81
    
    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

    # PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_captions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81