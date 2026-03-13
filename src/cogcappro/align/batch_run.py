"""Batch runner for the refactored align runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


def list_all_jobs(root: str):
    jobs = []
    for setting in ["intra-subject", "inter-subject"]:
        for exp_dir in Path(root).glob(f"{setting}*"):
            if not exp_dir.is_dir():
                continue
            for sub_dir in exp_dir.glob("sub-*_seed*"):
                match = re.match(r"(sub-\d+)_seed(\d+)", sub_dir.name)
                if match:
                    jobs.append((str(sub_dir), match.group(1), int(match.group(2))))
    return jobs


def run_one(args, exp_dir: str, subject: str, seed: int):
    del subject
    ckpt_name = (
        f"diffusion_model_best{args.output_suffix}.pth"
        if args.model_type == "diffusion"
        else "diffusion_model_best.pth"
    )
    ckpt = Path(exp_dir) / "diffusion_ckpt" / ckpt_name
    log_file = Path(exp_dir) / "run.log"

    if not args.overwrite and ckpt.exists():
        print(f"[Skip] {exp_dir} best checkpoint already exists.")
        return

    cmd = [
        sys.executable,
        "-m",
        f"{__package__}.main",
        "--exp_dir",
        exp_dir,
        "--seed",
        str(seed),
        "--epoch",
        str(args.epoch),
        "--lr",
        str(args.lr),
        "--device",
        str(args.device),
        "--model_type",
        args.model_type,
        "--output_suffix",
        args.output_suffix,
    ]

    print(f">>>> Launching {exp_dir}")
    with log_file.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="exp", help="Experiments root directory")
    parser.add_argument("--device", type=int, required=True, help="GPU number")
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true", help="Force rerun")
    parser.add_argument(
        "--model_type",
        type=str,
        default="diffusion",
        choices=["diffusion", "simple"],
        help="Model type: diffusion (DiffusionPipe) or simple (SimpleAlignPipe)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Output file suffix, e.g., '_original' produces diffusion_model_best_original.pth and generated_embeddings_original.pt",
    )
    args = parser.parse_args()

    jobs = list_all_jobs(args.root_dir)
    print(f"Found {len(jobs)} experiments under {args.root_dir}, device={args.device}, running sequentially...")
    for exp_dir, subject, seed in jobs:
        run_one(args, exp_dir, subject, seed)
    print("All experiments completed!")


if __name__ == "__main__":
    main()
