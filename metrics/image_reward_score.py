from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import numpy as np
import torch
import ImageReward as RM


def load_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def load_prompts(prompt_file: str | Path) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [l for l in lines if l]
    if not lines:
        raise ValueError("No prompts found in file")
    return lines


def compute_image_reward_scores(
    image_dir: str | Path,
    prompt_file: str | Path,
    device: str = "cuda"
):
    img_paths = load_image_paths(image_dir)
    prompts = load_prompts(prompt_file)

    if len(prompts) == 1:
        prompts = prompts * len(img_paths)

    if len(prompts) != len(img_paths):
        raise ValueError(
            f"Prompts ({len(prompts)}) and images ({len(img_paths)}) must match."
        )

    # Load ImageReward model
    scorer = RM.load("ImageReward-v1.0", torch.device(device))

    scores = []

    for img_path, prompt in zip(img_paths, prompts):
        image = Image.open(img_path).convert("RGB")
        score = scorer.score(prompt, image)

        print(f"{img_path.name}: IR = {score:.4f}")

        scores.append(score)

    scores = np.array(scores, dtype=float)

    mean = scores.mean()
    std = scores.std(ddof=1) if len(scores) > 1 else float("nan")

    print("\n====================================")
    if np.isnan(std):
        print(f"ImageReward = {mean:.4f} (only 1 image)")
    else:
        print(f"ImageReward = {mean:.4f} ± {std:.4f}")
    print("====================================\n")


if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1"
    PROMPT_FILE = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\prompt.txt"

    compute_image_reward_scores(
        IMAGE_DIR,
        PROMPT_FILE,
        device="cuda"
    )
