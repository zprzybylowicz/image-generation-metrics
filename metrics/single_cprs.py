from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFile
import open_clip
import joblib

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cuda",
):
    """
    Ładuje CLIP – musi być identyczny z tym, na którym trenowany był klasyfikator!
    """
    if device == "cuda" and torch.cuda.is_available():
        device_t = torch.device("cuda")
    else:
        device_t = torch.device("cpu")
        print("[WARN] CUDA niedostępna – używam CPU.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(device_t)
    model.eval()
    return model, preprocess, device_t


def get_clip_embedding(img_path: Path, model, preprocess, device_t) -> np.ndarray:
    """
    Zwraca L2-znormalizowany embedding obrazu z CLIP.
    """
    img = Image.open(img_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device_t)

    with torch.no_grad():
        emb = model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy()[0]


if __name__ == "__main__":

    IMAGE_PATH = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1\sdxl-turbo (1).png"

    MODEL_NAME = "ViT-L-14"      # musi być TEN SAM model co w treningu
    PRETRAINED = "openai"
    DEVICE = "cuda"

    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # -----------------------------------------------
    # 1. Wczytanie wytrenowanego klasyfikatora
    # -----------------------------------------------
    clf_path = Path("clip_realism_clf.joblib")
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Classifier file not found: {clf_path}. "
            f"Najpierw uruchom skrypt treningowy."
        )
    clf = joblib.load(clf_path)
    print("[INFO] Loaded classifier.")

    # -----------------------------------------------
    # 2. Ładowanie modelu CLIP
    # -----------------------------------------------
    print("[INFO] Loading CLIP model...")
    model, preprocess, device_t = load_clip_model(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )
    print("[INFO] CLIP loaded.")

    # -----------------------------------------------
    # 3. Liczenie embeddingu obrazu
    # -----------------------------------------------
    print("[INFO] Computing image embedding...")
    emb = get_clip_embedding(img_path, model, preprocess, device_t)

    # -----------------------------------------------
    # 4. Przewidywanie realizmu
    # -----------------------------------------------
    prob_real = float(clf.predict_proba([emb])[0, 1])

    print("\n====================================")
    print(f"Image: {img_path.name}")
    print(f"CLIP Realism Score (p(real)) = {prob_real:.4f}")
    print("------------------------------------")
    print("Interpretacja:")
    print("  ~1.0 → wygląda jak prawdziwe zdjęcie")
    print("  ~0.5 → graniczne / mieszane cechy")
    print("  ~0.0 → wygląda jak obraz generowany")
    print("====================================\n")
