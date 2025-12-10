from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
import numpy as np
import open_clip


def load_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = [".png", ".jpg", ".jpeg"]
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def load_prompts(prompt_file: str | Path) -> List[str]:
    prompt_file = Path(prompt_file)
    if not prompt_file.is_file():
        raise ValueError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # usuń puste linie
    lines = [l for l in lines if l]
    if not lines:
        raise ValueError("No prompts found in file")
    return lines


def compute_clip_score(
    image_dir: str | Path,
    prompt_file: str | Path,
    device: str = "cuda",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> Tuple[float, float]:
    """
    Zwraca (mean_clip, std_clip), gdzie:
      - mean_clip: średnie cosinusowe podobieństwo obraz–tekst
      - std_clip: odchylenie standardowe; dla 1 obrazu zwraca NaN
    """
    # 1. wczytujemy ścieżki do obrazów i prompty
    img_paths = load_image_paths(image_dir)
    prompts = load_prompts(prompt_file)

    if len(prompts) == 1:
        # jeden wspólny prompt dla wszystkich obrazów
        prompts = prompts * len(img_paths)

    if len(prompts) != len(img_paths):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) does not match number of images ({len(img_paths)})"
        )

    # 2. ładujemy model CLIP
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA niedostępna – używam CPU.")
        device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device_t)
    model.eval()

    sims: List[float] = []

    # 3. pętla po parach (obraz, tekst)
    with torch.no_grad():
        for img_path, text in zip(img_paths, prompts):
            # obraz -> embedding
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device_t)  # [1, 3, H, W]

            # tekst -> embedding
            text_tokens = tokenizer([text]).to(device_t)  # [1, L]

            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(text_tokens)

            # normalizacja L2
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            # cosinusowe podobieństwo
            sim = (img_feat * txt_feat).sum(dim=-1)  # [1]
            sims.append(float(sim.item()))

    sims = np.array(sims, dtype=np.float32)

    # --- NORMALIZACJA DO [0, 1] ---
    sims = (sims + 1.0) / 2.0  # teraz każda wartość jest w zakresie [0, 1]

    mean = float(sims.mean())

    if len(sims) > 1:
        std = float(sims.std(ddof=1))
    else:
        std = float("nan")

    return mean, std


if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1"
    PROMPT_FILE = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\prompt.txt"
    DEVICE = "cuda"

    mean_clip, std_clip = compute_clip_score(
        IMAGE_DIR,
        PROMPT_FILE,
        device=DEVICE,
        model_name="ViT-B-32",
        pretrained="openai",
    )

    if np.isnan(std_clip):
        print(f"CLIP SCORE = {mean_clip:.4f} (tylko 1 obraz – brak odchylenia)")
    else:
        print(f"CLIP SCORE = {mean_clip:.4f} ± {std_clip:.4f}")
