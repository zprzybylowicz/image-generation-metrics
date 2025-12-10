import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


# ==============================
# HELPERY GEOMETRYCZNE
# ==============================

def joint_angle(a, b, c):
    """
    Oblicza kąt ABC (w stopniach)
    a, b, c — punkty 2D (np.array([x,y]))
    """
    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def dist(a, b):
    """Euklidesowa odległość między punktami 2D."""
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


# ==============================
# MEDIAPIPE – LANDMARKI POZY I DŁONI
# ==============================

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def extract_landmarks(image_bgr):
    """
    Zwraca słownik {nazwa_punktu: np.array([x,y])} w normalizacji [0,1]
    dla pojedynczej osoby (cały obraz = jedna osoba).
    """
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        points = {}
        for name, idx in POSE_LANDMARKS.items():
            p = lm[idx]
            # używamy tylko pewnych punktów
            if p.visibility < 0.5:
                points[name] = None
            else:
                points[name] = np.array([p.x, p.y], dtype=np.float32)

        return points


def count_visible_hands(image_bgr):
    """
    Zwraca przybliżoną liczbę dłoni widocznych w obrazie osoby.
    Używa MediaPipe Hands (0, 1 lub 2 dłonie).
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
    ) as hands:

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return 0

        return min(len(results.multi_hand_landmarks), 2)


# ==============================
# OCENA KOŃCZYNY
# ==============================

def evaluate_limb(p1, p2, p3):
    """
    Kończyna: np. bark–łokieć–nadgarstek.
    Zwraca:
      - wartość 0–1 jeśli kończynę da się ocenić,
      - None jeśli kończyna niewidoczna (np. zasłonięta).
    """
    d1 = dist(p1, p2)
    d2 = dist(p2, p3)

    # jeśli brak któregoś punktu → nie oceniamy tej kończyny
    if d1 is None or d2 is None:
        return None

    # --- 1. Ciągłość ---
    continuity = 1.0 if (d1 > 0.001 and d2 > 0.001) else 0.0

    # --- 2. Proporcje segmentów ---
    if d2 == 0:
        prop_score = 0.0
    else:
        ratio = d1 / d2
        # ostre progi
        if 0.75 <= ratio <= 1.25:
            prop_score = 1.0
        elif 0.6 <= ratio <= 1.4:
            prop_score = 0.5
        else:
            prop_score = 0.1

    # --- 3. Kąt w stawie ---
    angle = joint_angle(p1, p2, p3)
    if angle is None:
        angle_score = 0.1
    else:
        if 70 <= angle <= 140:
            angle_score = 1.0        # naturalne zgięcie
        elif 45 <= angle < 70 or 140 < angle <= 160:
            angle_score = 0.5        # trochę sztywno / przeprost
        else:
            angle_score = 0.1        # bardzo nienaturalne

    # --- 4. Łączny wynik kończyny ---
    limb_score = 0.5 * continuity + 0.3 * angle_score + 0.2 * prop_score
    return float(np.clip(limb_score, 0.0, 1.0))


# ==============================
# LIMBQ DLA JEDNEJ OSOBY
# ==============================

def limb_single_person(image_bgr):
    """
    LimbQ dla pojedynczej osoby (obraz zawiera jedną osobę).
    Kończyny niewidoczne są pomijane w średniej,
    dodatkowo uwzględniamy liczbę widocznych dłoni.
    """
    landmarks = extract_landmarks(image_bgr)
    if landmarks is None:
        return 0.0

    limb_scores = []

    def add_limb(p1, p2, p3):
        s = evaluate_limb(p1, p2, p3)
        if s is not None:
            limb_scores.append(s)

    # ręce
    add_limb(landmarks["left_shoulder"],  landmarks["left_elbow"],  landmarks["left_wrist"])
    add_limb(landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"])
    # nogi
    add_limb(landmarks["left_hip"],  landmarks["left_knee"],  landmarks["left_ankle"])
    add_limb(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])

    # jeśli nic nie było widoczne → 0
    if len(limb_scores) == 0:
        return 0.0

    base_score = float(np.mean(limb_scores))

    # --- łagodna kara za dużą liczbę braków landmarków ---
    none_count = sum(1 for v in landmarks.values() if v is None)
    missing_penalty = 1.0
    if none_count >= 4:
        missing_penalty = 0.9
    if none_count >= 8:
        missing_penalty = 0.7

    # --- kara za brak dłoni ---
    num_hands = count_visible_hands(image_bgr)
    if num_hands >= 2:
        hand_penalty = 1.0
    elif num_hands == 1:
        hand_penalty = 0.7
    else:   # 0 dłoni
        hand_penalty = 0.4

    final_score = base_score * missing_penalty * hand_penalty
    return float(np.clip(final_score, 0.0, 1.0))


# ==============================
# YOLO – DETEKCJA OSÓB
# ==============================

# dla lepszej jakości możesz zmienić na "yolov8s.pt"
yolo_model = YOLO("yolov8n.pt")


def detect_person_boxes(image_bgr, conf_th=0.2, scale=1.0):
    """
    Zwraca listę bounding boxów [x1, y1, x2, y2] dla klasy 'person'
    w współrzędnych pikselowych.
    Można podbić scale > 1.0, żeby YOLO widział małe osoby.
    """
    h, w = image_bgr.shape[:2]

    if scale != 1.0:
        large = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LINEAR)
    else:
        large = image_bgr

    results = yolo_model(large, verbose=False)
    boxes = []

    r = results[0]
    if r.boxes is None:
        return boxes

    for box in r.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id == 0 and conf >= conf_th:   # 'person'
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # skalujemy z powrotem do oryginalnego rozmiaru
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes


# ==============================
# MULTI-PERSON + WIZUALIZACJA
# ==============================

def compute_limbq_multi_with_vis(image_path,
                                 min_person_size=32,
                                 conf_th=0.2,
                                 scale=1.0,
                                 save_path=None,
                                 show_window=False):
    """
    1. Wczytuje obraz
    2. Detekcja ludzi (YOLO)
    3. Dla każdej osoby:
        - crop
        - LimbQ
        - bbox + napis "LimbQ: x.xx"
    4. Zwraca (globalny_LimbQ, lista_score_per_person)
    5. Opcjonalnie zapisuje obraz z wizualizacją.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]

    boxes = detect_person_boxes(image, conf_th=conf_th, scale=scale)
    person_scores = []

    for (x1, y1, x2, y2) in boxes:
        pw, ph = x2 - x1, y2 - y1
        if pw < min_person_size or ph < min_person_size:
            continue

        pad = int(0.05 * max(pw, ph))
        xx1 = max(0, x1 - pad)
        yy1 = max(0, y1 - pad)
        xx2 = min(w, x2 + pad)
        yy2 = min(h, y2 + pad)

        person_crop = image[yy1:yy2, xx1:xx2]

        score = limb_single_person(person_crop)
        person_scores.append(score)

        # kolor: zielony dla "dobrych", czerwony dla słabych
        color = (0, 255, 0) if score >= 0.6 else (0, 0, 255)
        cv2.rectangle(image, (xx1, yy1), (xx2, yy2), color, 2)

        label = f"LimbQ: {score:.2f}"
        cv2.putText(image, label, (xx1, max(yy1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # fallback: nie wykryto ludzi lub za mali → licz na całym obrazie
    if len(person_scores) == 0:
        score = limb_single_person(image)
        person_scores.append(score)
        label = f"LimbQ: {score:.2f}"
        cv2.putText(image, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    global_score = float(np.mean(person_scores))

    if save_path is not None:
        cv2.imwrite(save_path, image)

    if show_window:
        cv2.imshow("LimbQ visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return global_score, person_scores


# ==============================
# PRZYKŁADOWE URUCHOMIENIE
# ==============================
import os


def process_folder(folder_path, output_folder,
                   min_person_size=32,
                   conf_th=0.2,
                   scale=1.5):

    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, f"{filename}_limbq.png")

        print(f"[INFO] Przetwarzam: {filename}")

        global_score, per_person = compute_limbq_multi_with_vis(
            input_path,
            min_person_size=min_person_size,
            conf_th=conf_th,
            scale=scale,
            save_path=output_path,
            show_window=True   # <-- tu jest podgląd dla KAŻDEGO obrazu
        )

        results.append({
            "filename": filename,
            "global_limbq": global_score,
            "per_person_scores": per_person
        })

    return results

if __name__ == "__main__":
    folder_in = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1"
    folder_out = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1_limbq"

    results = process_folder(
        folder_path=folder_in,
        output_folder=folder_out,
        min_person_size=32,
        conf_th=0.2,
        scale=1.5
    )

    print("\n[INFO] Przetwarzanie zakończone.")
    print(f"Wizualizacje zapisane w: {folder_out}")

