import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from calculate_fvd import calculate_fvd
from calculate_ssim import calculate_ssim


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(video_dir, recursive=False):
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in Path(video_dir).glob(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_gt_index(data_root):
    gt_index = {}
    for source in ("TII", "UZH"):
        video_dir = Path(data_root) / source / "videos"
        if not video_dir.exists():
            continue
        for path in find_videos(video_dir, recursive=False):
            gt_index.setdefault(path.stem, path)
    return gt_index


def load_video(path, num_frames, size):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = []
    try:
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"could not read frames from {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    array = np.asarray(frames[:num_frames], dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Compute FVD and SSIM for generated videos matched to AeroBench GT videos by filename."
    )
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--method", choices=("styleganv", "videogpt"), default="styleganv")
    parser.add_argument("--recursive", action="store_true", help="search generated_dir recursively")
    parser.add_argument("--per_frame", action="store_true", help="return per-frame SSIM and per-prefix FVD")
    parser.add_argument("--max_pairs", type=int, default=None, help="evaluate at most this many matched pairs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    gt_index = build_gt_index(args.data_root)
    generated_videos = find_videos(generated_dir, recursive=args.recursive)

    pairs = []
    missing = []
    for gen_path in generated_videos:
        gt_path = gt_index.get(gen_path.stem)
        if gt_path is None:
            missing.append(str(gen_path))
            continue
        pairs.append((gen_path, gt_path))

    if not pairs:
        raise SystemExit(
            f"No matching videos found. Checked {len(generated_videos)} generated videos "
            f"against {len(gt_index)} AeroBench GT videos."
        )

    total_pairs = len(pairs)
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise SystemExit("--max_pairs must be greater than 0")
        pairs = pairs[:args.max_pairs]

    videos_gen = torch.empty(
        len(pairs), args.frames, 3, args.size, args.size, dtype=torch.float32, requires_grad=False
    )
    videos_gt = torch.empty(
        len(pairs), args.frames, 3, args.size, args.size, dtype=torch.float32, requires_grad=False
    )
    matched_names = []

    print(f"Matched {total_pairs} generated videos to AeroBench GT videos.")
    if len(pairs) < total_pairs:
        print(f"Evaluating first {len(pairs)} matched pairs due to --max_pairs.")
    if missing:
        print(f"Skipping {len(missing)} generated videos with no GT match.")

    for idx, (gen_path, gt_path) in enumerate(tqdm(pairs, desc="loading videos")):
        videos_gen[idx] = load_video(gen_path, args.frames, args.size)
        videos_gt[idx] = load_video(gt_path, args.frames, args.size)
        matched_names.append(gen_path.stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "generated_dir": str(generated_dir),
        "data_root": str(Path(args.data_root)),
        "num_pairs": len(pairs),
        "total_matched_pairs": total_pairs,
        "frames": args.frames,
        "size": args.size,
        "matched_names": matched_names,
        "missing_generated": missing,
        "fvd": calculate_fvd(
            videos_gen,
            videos_gt,
            device,
            method=args.method,
            only_final=not args.per_frame,
        ),
        "ssim": calculate_ssim(videos_gen, videos_gt, only_final=not args.per_frame),
    }

    output = args.output
    if output is None:
        output = generated_dir / f"visual_metrics_{generated_dir.name}.json"
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as handle:
        json.dump(to_jsonable(result), handle, indent=4)

    print(json.dumps(to_jsonable(result), indent=4))
    print(f"Wrote metrics to {output}")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")
    main()
