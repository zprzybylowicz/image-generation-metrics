import os
import csv
from statistics import mean
import torch
import torchvision.transforms as T
from torchvision import models

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# (opcjonalnie) XGBoost – jeśli nie masz biblioteki, ten fragment się po prostu pominie
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ==============================
# PARAMETRY ŚCIEŻEK / TRYB PRACY
# ==============================

# Tryby:
# "extract"    – generuje cechy do FEATURES_CSV
# "train"      – trenuje model na LABELED_CSV (z kolumną 'label')
# "visualize"  – wizualizuje wynik modelu na jednym obrazku
# "debug_one"  – pokazuje ID osób dla VIS_IMAGE
# "debug_all"  – pokazuje ID osób dla wszystkich obrazków w FOLDER_IN (po kolei)
MODE = "visualize"  # "extract" / "train" / "visualize" / "debug_one" / "debug_all"

BASE_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics"

FOLDER_IN = os.path.join(BASE_DIR, "data", "g1")
FEATURES_CSV = os.path.join(BASE_DIR, "data", "g1_features.csv")
LABELED_CSV = os.path.join(BASE_DIR, "data", "g1_features_labeled.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "limbq_ml_model.pkl")
VIS_IMAGE = os.path.join(
    r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1\4.png"
)
VIS_OUT = os.path.join(BASE_DIR, "data", "g1_ml_vis.png")

# stałe do YOLO – ważne, żeby były takie same w extract/train/visualize/debug
MIN_PERSON_SIZE = 5
CONF_TH = 0.5
SCALE = 1.5


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
# MEDIAPIPE – POZA + DŁONIE
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

def load_artifact_cnn(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    return model, device, transform

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
            if p.visibility < 0.5:
                points[name] = None
            else:
                points[name] = np.array([p.x, p.y], dtype=np.float32)

        return points


def count_visible_hands(image_bgr):
    """
    Zwraca przybliżoną liczbę dłoni widocznych w obrazie osoby (0,1,2).
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

def predict_artifact_score(model, device, transform, crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # zakładamy cl1 = artifact
    return float(probs[1])

# ==============================
# OCENA KOŃCZYNY (rule-based)
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

    if d1 is None or d2 is None:
        return None

    # 1. ciągłość
    continuity = 1.0 if (d1 > 0.001 and d2 > 0.001) else 0.0

    # 2. proporcje
    if d2 == 0:
        prop_score = 0.0
    else:
        ratio = d1 / d2
        if 0.75 <= ratio <= 1.25:
            prop_score = 1.0
        elif 0.6 <= ratio <= 1.4:
            prop_score = 0.5
        else:
            prop_score = 0.1

    # 3. kąt
    angle = joint_angle(p1, p2, p3)
    if angle is None:
        angle_score = 0.1
    else:
        if 70 <= angle <= 140:
            angle_score = 1.0
        elif 45 <= angle < 70 or 140 < angle <= 160:
            angle_score = 0.5
        else:
            angle_score = 0.1

    limb_score = 0.5 * continuity + 0.3 * angle_score + 0.2 * prop_score
    return float(np.clip(limb_score, 0.0, 1.0))


# ==============================
# YOLO – DETEKCJA OSÓB
# ==============================

yolo_model = YOLO("yolov8n.pt")   # możesz zmienić na "yolov8s.pt"


def detect_person_boxes(image_bgr, conf_th=0.2, scale=1.0):
    """
    Zwraca listę bounding boxów [x1, y1, x2, y2] dla klasy 'person'.
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
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes


# ==============================
# CECHY DLA JEDNEJ OSOBY (LimbQ-ML)
# ==============================

def extract_person_features_from_crop(person_crop, bbox, image_shape):
    """
    Zwraca słownik cech dla jednej osoby:
    - średni / min / max wynik kończyn
    - liczba widocznych kończyn
    - liczba brakujących landmarków
    - liczba dłoni
    - wysokość i szerokość osoby względem obrazu
    """
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox
    box_h = y2 - y1
    box_w = x2 - x1

    box_height_norm = box_h / float(h_img)
    box_width_norm = box_w / float(w_img)
    box_area_norm = (box_w * box_h) / float(w_img * h_img)

    landmarks = extract_landmarks(person_crop)
    if landmarks is None:
        return None

    limb_scores = []

    def add_limb(p1, p2, p3):
        s = evaluate_limb(p1, p2, p3)
        if s is not None:
            limb_scores.append(s)

    # ręce
    add_limb(landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"])
    add_limb(landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"])
    # nogi
    add_limb(landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"])
    add_limb(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])

    visible_limbs = len(limb_scores)
    none_count = sum(1 for v in landmarks.values() if v is None)
    num_hands = count_visible_hands(person_crop)

    if visible_limbs > 0:
        mean_limb = mean(limb_scores)
        min_limb = min(limb_scores)
        max_limb = max(limb_scores)
    else:
        mean_limb = 0.0
        min_limb = 0.0
        max_limb = 0.0

    return {
        "mean_limb_score": float(mean_limb),
        "min_limb_score": float(min_limb),
        "max_limb_score": float(max_limb),
        "visible_limbs": int(visible_limbs),
        "none_count": int(none_count),
        "num_hands": int(num_hands),
        "box_height_norm": float(box_height_norm),
        "box_width_norm": float(box_width_norm),
        "box_area_norm": float(box_area_norm),
    }


FEATURE_COLS = [
    "mean_limb_score",
    "min_limb_score",
    "max_limb_score",
    "visible_limbs",
    "none_count",
    "num_hands",
    "box_height_norm",
    "box_width_norm",
    "box_area_norm",
]


# ==============================
# DEBUG: ID osób na jednym obrazku
# ==============================

def debug_show_person_ids(image_path,
                          min_person_size=MIN_PERSON_SIZE,
                          conf_th=CONF_TH,
                          scale=SCALE):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]
    boxes = detect_person_boxes(image, conf_th=conf_th, scale=scale)

    print(f"[DEBUG] {os.path.basename(image_path)}: wykryto {len(boxes)} osób")

    for pid, (x1, y1, x2, y2) in enumerate(boxes):
        pw, ph = x2 - x1, y2 - y1
        if pw < min_person_size or ph < min_person_size:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(image, f"id={pid}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.imshow("Person IDs", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_show_ids_for_all_images(folder_path,
                                  min_person_size=MIN_PERSON_SIZE,
                                  conf_th=CONF_TH,
                                  scale=SCALE):
    """
    Dla KAŻDEGO obrazka w folderze:
      - pokazuje ludzi z bboxami i id=person_id
      - sterowanie: dowolny klawisz = następny obraz, 'q' = wyjście
    """
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    files.sort()

    print(f"[DEBUG_ALL] Łącznie obrazków: {len(files)}")
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[DEBUG_ALL] Nie mogę wczytać: {filename}, pomijam.")
            continue

        h, w = image.shape[:2]
        boxes = detect_person_boxes(image, conf_th=conf_th, scale=scale)

        print(f"[DEBUG_ALL] {filename}: wykryto {len(boxes)} osób")

        for pid, (x1, y1, x2, y2) in enumerate(boxes):
            pw, ph = x2 - x1, y2 - y1
            if pw < min_person_size or ph < min_person_size:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(image, f"id={pid}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        cv2.imshow("Person IDs - " + filename, image)
        print("→ dowolny klawisz = następny obraz, 'q' = wyjście")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('q'):
            print("[DEBUG_ALL] przerwano przez użytkownika")
            break


# ==============================
# KROK 1: EKSTRAKCJA CECH DLA FOLDERU
# ==============================

def run_extract_features(folder_path, csv_out,
                         min_person_size=MIN_PERSON_SIZE,
                         conf_th=CONF_TH,
                         scale=SCALE):

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_filename", "person_id"] + FEATURE_COLS
        )

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue

            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            boxes = detect_person_boxes(image, conf_th=conf_th, scale=scale)
            print(f"[FEAT] {filename}: znaleziono {len(boxes)} osób")

            for pid, (x1, y1, x2, y2) in enumerate(boxes):
                pw, ph = x2 - x1, y2 - y1
                if pw < min_person_size or ph < min_person_size:
                    continue

                pad = int(0.05 * max(pw, ph))
                xx1 = max(0, x1 - pad)
                yy1 = max(0, y1 - pad)
                xx2 = min(w, x2 + pad)
                yy2 = min(h, y2 + pad)

                person_crop = image[yy1:yy2, xx1:xx2]

                feats = extract_person_features_from_crop(
                    person_crop, (xx1, yy1, xx2, yy2), image.shape
                )
                if feats is None:
                    continue

                row = [filename, pid] + [feats[c] for c in FEATURE_COLS]
                writer.writerow(row)

    print(f"[FEAT] Zapisano cechy do: {csv_out}")
    print("Dodaj teraz kolumnę 'label' (0/1) ręcznie (np. w Excelu).")


# ==============================
# KROK 2: TRENING MODELU LIMBQ-ML (ML + GridSearch)
# ==============================

def run_train_model(labeled_csv, model_out):
    # 1. Wczytanie danych
    df = pd.read_csv(labeled_csv, sep=';')

    if "label" not in df.columns:
        raise ValueError("W CSV musi być kolumna 'label' (0/1).")

    # zamiana label na liczby + wywalenie innych wartości
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()

    print(f"[TRAIN] Używam {len(df)} przykładów z oznaczonym label (0/1).")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # 2. Podział na trening / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Definicja modeli + przestrzeni hiperparametrów

    # 3.1 Random Forest (z class_weight='balanced')
    pipe_rf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid_rf = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": ["balanced"],
    }

    # 3.2 Gradient Boosting
    pipe_gb = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    )

    param_grid_gb = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    }

    model_specs = [
        ("RandomForest", pipe_rf, param_grid_rf),
        ("GradBoost", pipe_gb, param_grid_gb),
    ]

    # 3.3 (opcjonalnie) XGBoost – jeśli zainstalowany
    if HAS_XGB:
        pipe_xgb = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        param_grid_xgb = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__scale_pos_weight": [1.0],
        }

        model_specs.append(("XGBoost", pipe_xgb, param_grid_xgb))

    # 4. GridSearchCV dla każdego modelu – wybór najlepszego ROC-AUC
    best_model = None
    best_name = None
    best_auc = -1.0

    for name, pipe, param_grid in model_specs:
        print(f"\n[GRID] Trenuję model: {name}")
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        print(f"[GRID] {name} – najlepsze parametry: {grid.best_params_}")
        print(f"[GRID] {name} – średni ROC-AUC (CV): {grid.best_score_:.4f}")

        # ocena na test
        y_proba_test = grid.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_proba_test)
        print(f"[GRID] {name} – ROC-AUC (test): {auc_test:.4f}")

        if auc_test > best_auc:
            best_auc = auc_test
            best_model = grid.best_estimator_
            best_name = name

    # 5. Ewaluacja najlepszego modelu
    print("\n=== NAJLEPSZY MODEL ===")
    print("Model:", best_name)
    print("ROC-AUC (test):", best_auc)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Classification report (najlepszy model) ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC (test):", roc_auc_score(y_test, y_proba))

    # 6. Zapis modelu
    joblib.dump(best_model, model_out)
    print("Zapisano model do:", model_out)


# ==============================
# KROK 3: WIZUALIZACJA LIMBQ-ML NA OBRAZKU
# ==============================
def run_visualize_ml(image_path, model_path, out_path,
                     min_person_size=MIN_PERSON_SIZE,
                     conf_th=CONF_TH,
                     scale=SCALE):

    # ---- 1. Wczytanie modeli ----
    clf = joblib.load(model_path)  # LimbQ-ML
    artifact_model, art_device, art_transform = load_artifact_cnn(
        os.path.join(BASE_DIR, "data", "artifact_cnn_resnet18.pt")
    )

    # ---- 2. Wczytanie obrazu ----
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]
    boxes = detect_person_boxes(image, conf_th=conf_th, scale=scale)

    person_scores = []  # końcowe HAS dla każdego człowieka

    # ---- 3. Przetwarzanie każdej osoby ----
    for (x1, y1, x2, y2) in boxes:
        pw, ph = x2 - x1, y2 - y1
        if pw < min_person_size or ph < min_person_size:
            continue

        # lekki padding
        pad = int(0.05 * max(pw, ph))
        xx1 = max(0, x1 - pad)
        yy1 = max(0, y1 - pad)
        xx2 = min(w, x2 + pad)
        yy2 = min(h, y2 + pad)

        person_crop = image[yy1:yy2, xx1:xx2]

        # ---- LimbQ-ML cechy ----
        feats = extract_person_features_from_crop(
            person_crop, (xx1, yy1, xx2, yy2), image.shape
        )
        if feats is None:
            continue

        x = [[feats[c] for c in FEATURE_COLS]]
        limbq_good = float(clf.predict_proba(x)[0, 1])      # P(poprawna poza)
        limbq_bad  = 1.0 - limbq_good                      # anomalia poza

        # ---- ArtifactCNN ----
        art_score = predict_artifact_score(artifact_model, art_device, art_transform, person_crop)
        # art_score = 0 (ok) → 1 (artefakt)

        # ---- Final HAS ----
        has_score = 0.6 * art_score + 0.4 * limbq_bad
        person_scores.append(has_score)

        # ---- Kolor: im więcej artefaktów, tym bardziej czerwony ----
        if has_score < 0.1:
            color = (0, 255, 0)       # zielony
        elif has_score < 0.4:
            color = (0, 255, 255)     # żółty
        else:
            color = (0, 0, 255)       # czerwony

        label = f"A:{art_score:.2f}  L:{limbq_bad:.2f}  HAS:{has_score:.2f}"

        cv2.rectangle(image, (xx1, yy1), (xx2, yy2), color, 2)
        cv2.putText(image, label, (xx1, max(yy1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # ---- 4. Wynik globalny ----
    if len(person_scores) > 0:
        global_score = float(np.mean(person_scores))
    else:
        global_score = 0.0

    # ---- 5. Zapisywanie i podsumowanie ----
    cv2.imwrite(out_path, image)
    cv2.imshow("HAS (Human Artifact Score)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Global HAS:", global_score)
    print("Per person:", person_scores)
    print("Zapisano wizualizację do:", out_path)



# ==============================
# GŁÓWNY PRZEŁĄCZNIK
# ==============================

if __name__ == "__main__":
    if MODE == "extract":
        run_extract_features(FOLDER_IN, FEATURES_CSV)

    elif MODE == "train":
        run_train_model(LABELED_CSV, MODEL_PATH)

    elif MODE == "visualize":
        run_visualize_ml(VIS_IMAGE, MODEL_PATH, VIS_OUT)

    elif MODE == "debug_one":
        debug_show_person_ids(VIS_IMAGE)

    elif MODE == "debug_all":
        debug_show_ids_for_all_images(FOLDER_IN)

    else:
        print("Nieznany MODE. Ustaw MODE na: 'extract', 'train', 'visualize', 'debug_one' albo 'debug_all'.")
