from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
import open_clip
from sklearn.linear_model import LogisticRegression
import joblib

# Pozwala PIL wczytywać częściowo uszkodzone pliki
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================
# 🔧 funkcja logowania progresu
# ==============================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def log_progress(current: int, total: int, prefix: str = ""):
    pct = int((current / total) * 100)
    if pct % 5 == 0:  # log co 5%
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix} {pct}% ({current}/{total})", end="\r")


# ==============================
# 🔧 wczytywanie listy obrazów
# ==============================
def load_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


# ==============================
# 🔧 ładowanie CLIP
# ==============================
def load_clip_model(
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
    device: str = "cuda",
):
    if device == "cuda" and torch.cuda.is_available():
        device_t = torch.device("cuda")
        log("Using CUDA GPU")
    else:
        device_t = torch.device("cpu")
        log("CUDA not available — using CPU")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(device_t)
    model.eval()
    log(f"Loaded CLIP model: {model_name} ({pretrained})")
    return model, preprocess, device_t


def compute_embeddings_for_paths(
    paths: List[Path],
    model,
    preprocess,
    device_t: torch.device,
    label: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Liczy embeddingi CLIP dla listy ścieżek w batchach.

    Zwraca:
      - X: macierz [N, D] z embeddingami
      - y: wektor etykiet [N] (wszędzie 'label')
      - ignored: liczba pominiętych plików (uszkodzonych / nie do wczytania)
    """
    total = len(paths)
    log(f"Starting embedding computation ({total} images, batch={batch_size})")
    start_time = time.time()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    ignored = 0

    with torch.no_grad():
        i = 0
        batch_idx = 0

        while i < total:
            batch_paths = paths[i : i + batch_size]
            batch_idx += 1

            imgs_t = []

            # 1. Wczytujemy obrazy i preprocessujemy
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img_t = preprocess(img)
                    imgs_t.append(img_t)
                except Exception as e:
                    print(f"[WARN] Could not load {p}: {e}")
                    ignored += 1

            if imgs_t:
                imgs_batch = torch.stack(imgs_t, dim=0).to(device_t)
                feats = model.encode_image(imgs_batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy()

                for k in range(feats_np.shape[0]):
                    X_list.append(feats_np[k])
                    y_list.append(label)

            i += batch_size

            # 🔥 log po każdym batchu
            done = min(i, total)
            pct = 100.0 * done / total
            log(f"[BATCH {batch_idx}] processed {done}/{total} images ({pct:.1f}%)")

    elapsed = time.time() - start_time
    log(f"Finished embedding computation in {elapsed:.2f} seconds.")
    log(f"Ignored {ignored} corrupted/unreadable images.")

    if not X_list:
        raise RuntimeError("No valid images to embed – X_list is empty.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y, ignored


# ==============================
# 🔧 MAIN
# ==============================
if __name__ == "__main__":
    REAL_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\real"
    GEN_DIR  = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\generated"
    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    DEVICE = "cuda"

    log("Starting CLIP Realism Training Pipeline")

    # log system info
    import sys, sysconfig
    log(f"Python exe: {sys.executable}")
    log(f"Env path:   {sys.prefix}")
    log(f"Site-pack:  {sysconfig.get_paths()['purelib']}")

    # 1. Model CLIP
    model, preprocess, device_t = load_clip_model(MODEL_NAME, PRETRAINED, DEVICE)

    # 2. Wczytanie ścieżek
    real_paths = load_image_paths(REAL_DIR)
    gen_paths  = load_image_paths(GEN_DIR)

    log(f"Found {len(real_paths)} REAL images.")
    log(f"Found {len(gen_paths)} GENERATED images.")

    # 3. Balansowanie
    n = min(len(real_paths), len(gen_paths))
    real_paths = real_paths[:n]
    gen_paths  = gen_paths[:n]
    log(f"Balanced to {n} real + {n} generated images.")

    # 4. Embeddingi – REAL
    X_real, y_real, ignored_real = compute_embeddings_for_paths(
        real_paths, model, preprocess, device_t, label=1, batch_size=64
    )

    # 5. Embeddingi – GENERATED
    X_gen, y_gen, ignored_gen = compute_embeddings_for_paths(
        gen_paths, model, preprocess, device_t, label=0, batch_size=64
    )

    # 6. Łączenie zbioru
    X = np.concatenate([X_real, X_gen])
    y = np.concatenate([y_real, y_gen])

    log(f"Final dataset: {X.shape[0]} samples, feature_dim = {X.shape[1]}")
    log(f"Ignored {ignored_real} real + {ignored_gen} generated images.")

    # 7. Trening klasyfikatora
    log("Training LogisticRegression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)
    log("Training completed.")

    joblib.dump(clf, "clip_realism_clf.joblib")
    log("Saved model as clip_realism_clf.joblib")

    # 8. Ewaluacja
    probs = clf.predict_proba(X)[:, 1]
    real_mean = probs[y == 1].mean()
    gen_mean  = probs[y == 0].mean()

    log("=== Summary ===")
    log(f"Mean p(real) for REAL images:       {real_mean:.4f}")
    log(f"Mean p(real) for GENERATED images: {gen_mean:.4f}")
    log("================")
