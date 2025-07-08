# label_corrector.py (Versión 2.3 - Con Rutas Absolutas y Resumible)
import cv2
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURACIÓN ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'potentially_good_images')
PSEUDO_LABELS_DIR = os.path.join(PROJECT_ROOT, 'pseudo_labels')
FINAL_DATASET_DIR = os.path.join(PROJECT_ROOT, 'final_verified_dataset')

CLASSES = {
    '0': 'original', '1': 'nuez', '2': 'marmoleada', '3': 'chocolate'
}
CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
# --- FIN DE LA CONFIGURACIÓN ---

# ... (El resto de las funciones: load_yolo_labels, save_yolo_labels, run_label_corrector) ...
# El código es idéntico al del mensaje anterior y está diseñado para reanudar el trabajo.
def load_yolo_labels(label_path, img_shape):
    detections = []
    h, w = img_shape
    if not os.path.exists(label_path): return detections
    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            detections.append({'class_id': class_id, 'polygon': coords.astype(int)})
    return detections

def save_yolo_labels(label_path, img_shape, detections, img_path, images_output_dir):
    h, w = img_shape
    lines = []
    for det in detections:
        class_id = det['class_id']
        polygon_points = det['polygon'].flatten().tolist()
        normalized_points = [f"{pt / w if i % 2 == 0 else pt / h:.6f}" for i, pt in enumerate(polygon_points)]
        line = f"{class_id} " + " ".join(map(str, normalized_points))
        lines.append(line)
    
    if lines:
        dest_img_path = os.path.join(images_output_dir, os.path.basename(img_path))
        if not os.path.exists(dest_img_path):
            shutil.copy(img_path, dest_img_path)
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
        return True
    return False

def run_label_corrector():
    print("--- FASE 2: Herramienta de Corrección de Etiquetas (v2.3 Absoluta) ---")
    
    images_output_dir = os.path.join(FINAL_DATASET_DIR, 'images')
    labels_output_dir = os.path.join(FINAL_DATASET_DIR, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(PSEUDO_LABELS_DIR, '*.txt')))
    print(f"Se encontraron {len(label_files)} imágenes con etiquetas para revisar.")
    
    for label_path in tqdm(label_files, desc="Revisando Imágenes"):
        basename = os.path.splitext(os.path.basename(label_path))[0]
        
        final_label_path = os.path.join(labels_output_dir, f"{basename}.txt")
        if os.path.exists(final_label_path):
            continue

        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(ALL_IMAGES_DIR, f"{basename}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if not img_path: continue

        image = cv2.imread(img_path)
        if image is None: continue

        detections = load_yolo_labels(label_path, image.shape[:2])
        if not detections: continue

        while True:
            display_img = image.copy()
            
            for i, det in enumerate(detections):
                class_id = det['class_id']
                polygon = det['polygon']
                color = CLASS_COLORS[class_id] if 0 <= class_id < len(CLASS_COLORS) else (255, 255, 255)
                
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [polygon], color, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
                cv2.drawContours(display_img, [polygon], -1, color, 2)
                
                label_pos = tuple(polygon.mean(axis=0).astype(int))
                cv2.putText(display_img, str(i + 1), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(display_img, str(i + 1), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            h, w = display_img.shape[:2]
            if h > 900: display_img = cv2.resize(display_img, (int(w * 900 / h), 900))
            
            cv2.imshow('Corrector de Etiquetas', display_img)
            
            print("\n" + "="*50 + f"\nRevisando: {basename}")
            print("Detecciones Actuales:")
            for i, det in enumerate(detections):
                print(f"  [{i+1}] {CLASSES.get(str(det['class_id']), 'DESCONOCIDO')}")

            print("\nAcciones:\n  [s] Siguiente/Aceptar y Guardar   [b] Borrar Detección   [c] Corregir Clase   [q] Salir")
            
            key = cv2.waitKey(0)
            key_char = chr(key & 0xFF).lower()

            if key_char == 's':
                break
            elif key_char == 'b':
                try:
                    idx_to_del = int(input("  Número de detección a BORRAR: "))
                    if 1 <= idx_to_del <= len(detections):
                        detections.pop(idx_to_del - 1)
                except (ValueError, IndexError): print("  Entrada inválida.")
            elif key_char == 'c':
                try:
                    idx_to_corr = int(input("  Número de detección a CORREGIR: "))
                    if 1 <= idx_to_corr <= len(detections):
                        print("    Clases: 1=original, 2=nuez, 3=marmoleada, 4=chocolate")
                        new_cls_str = input(f"    Nueva clase para detección {idx_to_corr}: ")
                        if new_cls_str in ['1', '2', '3', '4']:
                            detections[idx_to_corr - 1]['class_id'] = int(new_cls_str) - 1
                        else: print("    Clase inválida.")
                except (ValueError, IndexError): print("  Entrada inválida.")
            elif key_char == 'q':
                cv2.destroyAllWindows()
                print("Guardando trabajo de la imagen actual antes de salir...")
                if save_yolo_labels(final_label_path, image.shape[:2], detections, img_path, images_output_dir):
                     print("Progreso guardado.")
                print("¡Hasta la próxima!")
                return
        
        cv2.destroyAllWindows()
        if not save_yolo_labels(final_label_path, image.shape[:2], detections, img_path, images_output_dir):
             print(f"  -> Todas las detecciones fueron eliminadas para {basename}.")

if __name__ == '__main__':
    run_label_corrector()