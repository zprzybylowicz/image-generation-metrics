from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile
import mediapipe as mp
import torch
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============== MediaPipe ==============
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.1,  # trochę niżej, żeby był bardziej czuły
)
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.1,
)


def load_image_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {path}")
    return img_bgr


def detect_person_mediapipe(
    image_path: str | Path,
    use_face: bool = True,
    use_pose: bool = True,
) -> Tuple[bool, bool, bool]:
    img_bgr = load_image_bgr(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    has_face = False
    has_pose_ = False

    if use_face:
        res_face = mp_face.process(img_rgb)
        if res_face.detections:
            has_face = True

    if use_pose:
        res_pose = mp_pose.process(img_rgb)
        if res_pose.pose_landmarks:
            has_pose_ = True

    has_person = bool(has_face or has_pose_)
    return has_person, has_face, has_pose_


# ============== CLIP person detector ==============

def load_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cuda",
):
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


def clip_person_prob(
    image_path: str | Path,
    model,
    preprocess,
    device_t: torch.device,
) -> float:
    """
    Zwraca prawdopodobieństwo obecności osoby na obrazie wg CLIP.
    """
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device_t)

    texts = [
        "a close-up photo of a person's face",
        "a full body photo of a person",
        "a group of people",
        "a photo with no people"
    ]
    with torch.no_grad():
        text_tokens = open_clip.tokenize(texts).to(device_t)
        img_feat = model.encode_image(img_t)
        text_feat = model.encode_text(text_tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logits = (img_feat @ text_feat.T)[0]  # [3]
        # softmax jako pseudo-prawdopodobieństwo
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [3]

    p_person = float(probs[0] +  probs[1] + probs[2]+probs[3]   )  # suma "person" + "people"
    return p_person


# ============== HYBRYDOWA FUNKCJA ==============

def detect_person_hybrid(
    image_path: str | Path,
    clip_model=None,
    clip_preprocess=None,
    clip_device=None,
    clip_threshold: float = 0.5,
) -> Tuple[bool, bool, bool, float]:
    """
    Zwraca:
      has_person, has_face, has_pose, p_person_clip

    Logika:
      1) Jeśli MediaPipe wykryje twarz lub pozę → has_person = True.
      2) Jeśli nie, używamy CLIP:
         - liczymy p_person_CLIP,
         - jeśli p_person_CLIP >= threshold → has_person = True.
    """
    # 1. MediaPipe
    has_person_mp, has_face, has_pose = detect_person_mediapipe(image_path)

    p_person_clip = 0.0
    has_person_final = has_person_mp

    # 2. Jeśli MediaPipe nie znalazł osoby, podpytaj CLIP
    if not has_person_mp:
        if clip_model is None:
            clip_model, clip_preprocess, clip_device = load_clip_model()

        p_person_clip = clip_person_prob(
            image_path, clip_model, clip_preprocess, clip_device
        )
        if p_person_clip >= clip_threshold:
            has_person_final = True

    return has_person_final, has_face, has_pose, p_person_clip


if __name__ == "__main__":
    TEST_IMAGE = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1\dreamshaperxl_lightning.png"

    print("[INFO] Loading CLIP...")
    model, preprocess, device_t = load_clip_model()

    print("[INFO] Running hybrid person detection...")
    has_person, has_face, has_pose, p_clip = detect_person_hybrid(
        TEST_IMAGE,
        clip_model=model,
        clip_preprocess=preprocess,
        clip_device=device_t,
        clip_threshold=0.5,
    )

    print(f"Image: {TEST_IMAGE}")
    print(f"has_person (hybrid) = {has_person}")
    print(f"  has_face (MediaPipe) = {has_face}")
    print(f"  has_pose (MediaPipe) = {has_pose}")
    print(f"  p_person_CLIP        = {p_clip:.4f}")
