from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# =========================
#  POMOCNICZE: ŁADOWANIE
# =========================

def load_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


# =========================
#  CZĘŚĆ 1: NSS / MSCN
# =========================

def compute_mscn(gray: np.ndarray, ksize: int = 7, sigma: float = 7/6) -> np.ndarray:
    """
    Liczy mapę MSCN (Mean Subtracted Contrast Normalization).
    Szarość w zakresie [0, 1].
    """
    mu = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    mu_sq = mu * mu
    sigma_sq = cv2.GaussianBlur(gray * gray, (ksize, ksize), sigma) - mu_sq
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-8))
    mscn = (gray - mu) / (sigma + 1e-8)
    return mscn


def extract_nss_features(mscn: np.ndarray) -> np.ndarray:
    """
    Proste cechy NSS:
    - średnia i wariancja MSCN
    - kurtoza
    - korelacje z 4 sąsiadami (prawo, dół, prawy-dół, lewy-dół)
    """
    mean = float(mscn.mean())
    var = float(mscn.var() + 1e-8)

    kurt = float(((mscn - mean) ** 4).mean() / (var ** 2))

    corrs = []
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in shifts:
        shifted = np.roll(mscn, shift=(-dy, -dx), axis=(0, 1))
        corr = float((mscn * shifted).mean())
        corrs.append(corr)

    feats = np.array([mean, var, kurt] + corrs, dtype=np.float32)
    return feats


def nss_features_to_quality(feats: np.ndarray) -> float:
    """
    Heurystyczne mapowanie cech NSS na jakość [0, 1].
    Naturalne zdjęcia ~ mean ~ 0, var ~ 1, kurt ~ 3, korelacje niewielkie.
    """
    mean, var, kurt, c1, c2, c3, c4 = feats

    loss = (
        abs(mean) +
        abs(var - 1.0) +
        abs(kurt - 3.0) / 3.0 +
        (abs(c1) + abs(c2) + abs(c3) + abs(c4)) / 2.0
    )

    q = float(np.exp(-loss))  # 0 < exp(-loss) <= 1
    return q


# =========================
#  CZĘŚĆ 2: OSTROŚĆ
# =========================

def sharpness_quality(gray: np.ndarray) -> float:
    """
    Jakość ostrości na podstawie wariancji Laplasjanu.
    Zamieniamy szarość z [0,1] (float32) na uint8 [0,255],
    bo cv2.Laplacian ma focha na kombinację float32 -> CV_64F.
    """
    # gray jest w [0,1] float32 – zamieniamy na uint8
    gray_u8 = (gray * 255.0).clip(0, 255).astype(np.uint8)

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    var_lap = float(lap.var())

    # Logarytmiczna normalizacja – typowe wartości ~[0, 4]
    score = np.log1p(var_lap) / 5.0
    score = float(np.clip(score, 0.0, 1.0))
    return score


# =========================
#  CZĘŚĆ 3: SZUM
# =========================

def noise_quality(gray: np.ndarray) -> float:
    """
    Prosta ocena szumu:
    - wyliczamy różnicę między obrazem a jego rozmytą wersją (high-pass).
    - im większe odchylenie std tej różnicy, tym więcej szumu / detali.
    Traktujemy: im WIĘCEJ szumu, tym GORZEJ (czysto technicznie).
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.0)
    high_freq = gray - blurred
    noise_std = float(high_freq.std())

    # Zakładamy, że "ok" to <= 0.05, wyżej coraz gorzej
    # Skala gaussowska
    score = float(np.exp(- (noise_std / 0.05) ** 2))
    score = float(np.clip(score, 0.0, 1.0))
    return score


# =========================
#  CZĘŚĆ 4: KONTRAST
# =========================

def contrast_quality(gray: np.ndarray) -> float:
    """
    Ocena kontrastu na podstawie std jasności.
    - zbyt niski kontrast (płaski obraz) -> źle,
    - ekstremalnie wysoki (przepalenia / czarne plamy) -> też źle.
    Przyjmujemy optimum ~0.20.
    """
    std_luma = float(gray.std())

    # optimum = 0.2, sigma = 0.1
    score = float(np.exp(- ((std_luma - 0.2) / 0.1) ** 2))
    score = float(np.clip(score, 0.0, 1.0))
    return score


# =========================
#  FINAŁ: METRYKA ZŁOŻONA
# =========================

def compute_improved_quality(image_dir: str | Path) -> Tuple[float, float]:
    """
    Liczy złożoną metrykę jakości dla wszystkich obrazów w katalogu.
    Wynik w skali 0–1, gdzie 1 = bardzo dobra jakość, 0 = bardzo słaba.
    """
    img_paths = load_image_paths(image_dir)

    qs: List[float] = []

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p}")
            continue

        # skala szarości 0–1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # NSS
        mscn = compute_mscn(gray)
        feats = extract_nss_features(mscn)
        q_nss = nss_features_to_quality(feats)

        # ostrość
        q_sharp = sharpness_quality(gray)

        # szum
        q_noise = noise_quality(gray)

        # kontrast
        q_contrast = contrast_quality(gray)

        # wagi
        w_nss = 0.4
        w_sharp = 0.3
        w_noise = 0.15
        w_contrast = 0.15
        w_sum = w_nss + w_sharp + w_noise + w_contrast

        q_final = (
            w_nss * q_nss +
            w_sharp * q_sharp +
            w_noise * q_noise +
            w_contrast * q_contrast
        ) / w_sum

        qs.append(q_final)

        print(
            f"{p.name}: "
            f"Q_final={q_final:.3f}  "
            f"(NSS={q_nss:.3f}, sharp={q_sharp:.3f}, noise={q_noise:.3f}, contrast={q_contrast:.3f})"
        )

    if not qs:
        raise ValueError("No valid images.")

    arr = np.array(qs, dtype=np.float32)
    mean_q = float(arr.mean())
    std_q = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")

    return mean_q, std_q


if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1"

    mean_q, std_q = compute_improved_quality(IMAGE_DIR)

    print("\n====================================")
    if np.isnan(std_q):
        print(f"IMPROVED QUALITY = {mean_q:.3f} (tylko 1 obraz)")
    else:
        print(f"IMPROVED QUALITY = {mean_q:.3f} ± {std_q:.3f}")
    print("====================================\n")
