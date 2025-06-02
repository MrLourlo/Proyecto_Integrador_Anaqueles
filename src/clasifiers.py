import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_object_patches(data_root, patch_size=(64, 64)):
    X, y = [], []
    scene_folders = sorted(os.listdir(data_root))
    for scene in tqdm(scene_folders, desc="Procesando escenas"):
        scene_path = os.path.join(data_root, scene)
        if not os.path.isdir(scene_path): continue
        for frame in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame)
            rgb_path = os.path.join(frame_path, "rgb.png")
            seg_path = os.path.join(frame_path, "segmentation.png")
            json_path = os.path.join(frame_path, "instance_class_map.json")
            if not (os.path.exists(rgb_path) and os.path.exists(seg_path) and os.path.exists(json_path)):
                continue
            rgb = cv2.imread(rgb_path)
            seg = cv2.imread(seg_path, 0)
            with open(json_path, "r") as f:
                id_to_class = json.load(f)

            for instance_id_str, class_name in id_to_class.items():
                instance_id = int(instance_id_str)
                mask = (seg == instance_id).astype(np.uint8)
                if cv2.countNonZero(mask) < 30:
                    continue
                x, y_, w, h = cv2.boundingRect(mask)
                obj_patch = rgb[y_:y_+h, x:x+w]
                if obj_patch.size == 0:
                    continue
                obj_patch = cv2.resize(obj_patch, patch_size)
                X.append(obj_patch)
                y.append(class_name)

    return np.array(X), np.array(y)

def main():
    data_root = "output_scenes"
    X, y = extract_object_patches(data_root)
    print(f"\n🎯 Total objetos: {len(X)} — Clases únicas: {set(y)}")

    if len(X) == 0:
        print("❌ No se encontraron objetos para clasificar. Verifica tus segmentaciones y mapeos.")
        return

    X_flat = X.reshape(len(X), -1)  # Aplanar para modelos clásicos
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True)
    }

    for name, model in models.items():
        print(f"\n🧠 Entrenando modelo: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
