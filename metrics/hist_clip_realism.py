from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
import open_clip
import joblib
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def load_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cuda",
):
    if device == "cuda" and torch.cuda.is_available():
        device_t = torch.device("cuda")
        log("Using CUDA GPU")
    else:
        device_t = torch.device("cpu")
        log("CUDA not available — using CPU")

    log(f"Loading CLIP model: {model_name} / {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(device_t)
    model.eval()
    log("CLIP model loaded.")
    return model, preprocess, device_t


def compute_probs_for_paths(
    paths: List[Path],
    model,
    preprocess,
    device_t: torch.device,
    clf,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """
    Dla listy obrazów:
      - liczy embeddingi CLIP w batchach
      - przepuszcza przez classifier
      - zwraca wektor p(real) i listę ścieżek jako stringi
    """
    total = len(paths)
    log(f"Computing realism scores for {total} images (batch={batch_size})")
    start_time = time.time()

    probs_list: list[float] = []
    path_strs: list[str] = []

    with torch.no_grad():
        i = 0
        batch_idx = 0

        while i < total:
            batch_paths = paths[i : i + batch_size]
            batch_idx += 1

            imgs_t = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img_t = preprocess(img)
                    imgs_t.append(img_t)
                    valid_paths.append(str(p))
                except Exception as e:
                    print(f"[WARN] Could not load {p}: {e}")

            if imgs_t:
                imgs_batch = torch.stack(imgs_t, dim=0).to(device_t)
                feats = model.encode_image(imgs_batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy()  # [B, D]

                # p(real) dla całego batcha
                probs_batch = clf.predict_proba(feats_np)[:, 1]  # [B]

                probs_list.extend(probs_batch.tolist())
                path_strs.extend(valid_paths)

            i += batch_size
            done = min(i, total)
            pct = 100.0 * done / total
            log(f"[BATCH {batch_idx}] {done}/{total} images ({pct:.1f}%)")

    elapsed = time.time() - start_time
    log(f"Finished probability computation in {elapsed:.2f} seconds.")

    return np.array(probs_list, dtype=float), path_strs


if __name__ == "__main__":
    # 🔧 ŚCIEŻKI DO ZBIORÓW (te same co w treningu)
    REAL_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\real"
    GEN_DIR  = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\generated"

    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"
    DEVICE = "cuda"

    log("=== CLIP Realism Histogram Script ===")

    # 1. Wczytujemy classifier
    clf_path = Path("clip_realism_clf.joblib")
    if not clf_path.exists():
        raise FileNotFoundError("clip_realism_clf.joblib not found – najpierw uruchom trening.")
    clf = joblib.load(clf_path)
    log("Loaded classifier clip_realism_clf.joblib")

    # 2. Ładujemy CLIP
    model, preprocess, device_t = load_clip_model(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )

    # 3. Wczytujemy ścieżki obrazów
    real_paths = load_image_paths(REAL_DIR)
    gen_paths  = load_image_paths(GEN_DIR)

    log(f"Found {len(real_paths)} REAL images.")
    log(f"Found {len(gen_paths)} GENERATED images.")

    # Opcjonalnie możesz ograniczyć liczbę do szybkiego testu, np. 2000:
    # real_paths = real_paths[:2000]
    # gen_paths  = gen_paths[:2000]

    # 4. Liczymy p(real) dla REAL i GENERATED
    probs_real, _ = compute_probs_for_paths(real_paths, model, preprocess, device_t, clf, batch_size=64)
    probs_gen,  _ = compute_probs_for_paths(gen_paths,  model, preprocess, device_t, clf, batch_size=64)

    log(f"REAL: mean p(real) = {probs_real.mean():.4f}, N = {len(probs_real)}")
    log(f"GEN:  mean p(real) = {probs_gen.mean():.4f}, N = {len(probs_gen)}")

    # 5. Rysujemy histogramy
    log("Plotting histograms...")
    plt.figure(figsize=(8, 5))
    plt.hist(probs_real, bins=30, range=(0, 1), alpha=0.6, label="REAL")
    plt.hist(probs_gen,  bins=30, range=(0, 1), alpha=0.6, label="GENERATED")
    plt.xlabel("p(real)")
    plt.ylabel("Liczba obrazów")
    plt.title("Rozkład CLIP Realism Score dla REAL i GENERATED")
    plt.legend()
    plt.tight_layout()
    plt.savefig("clip_realism_hist.png", dpi=300)
    log("Saved histogram to clip_realism_hist.png")
    plt.show()

    log("Done.")
