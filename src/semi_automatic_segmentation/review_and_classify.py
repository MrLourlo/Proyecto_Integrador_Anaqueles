# curate_and_classify_v2.py (con contador de clases en tiempo real)
import cv2
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURACIÓN ---
ALL_IMAGES_DIR = 'potentially_good_images' 
RAW_MASKS_DIR = 'mantecadas_dataset_labels_v9_aggressive_vertical/masks'
FINAL_MULTICLASS_DIR = 'mantecadas_multiclass_dataset'

# --- ¡NUEVA CLASE AÑADIDA! ---
CLASSES = {
    '1': 'original',
    '2': 'nuez',
    '3': 'marmoleada',
    '4': 'chocolate'  # <--- NUEVA
}
# --- FIN DE LA CONFIGURACIÓN ---

def get_base_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def parse_raw_mask_filename(mask_filename):
    parts = mask_filename.split('_mantecadas_')
    return parts[0] if parts else None

def curate_and_classify():
    images_output_dir = os.path.join(FINAL_MULTICLASS_DIR, 'images')
    labels_output_dir = os.path.join(FINAL_MULTICLASS_DIR, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    masks_by_image = defaultdict(list)
    for mask_path in sorted(glob.glob(os.path.join(RAW_MASKS_DIR, '*.png'))):
        img_basename = parse_raw_mask_filename(get_base_filename(mask_path))
        if img_basename:
            masks_by_image[img_basename].append(mask_path)
    
    print(f"Se encontraron {len(masks_by_image)} imágenes únicas para revisar.")

    # --- NUEVO: CONTADOR DE CLASES ---
    class_counts = defaultdict(int)
    # Cargar progreso previo si existe
    print("Calculando progreso de sesiones anteriores...")
    existing_labels = glob.glob(os.path.join(labels_output_dir, '*.txt'))
    for label_file in existing_labels:
        with open(label_file, 'r') as f:
            for line in f:
                class_index_str = line.split()[0]
                if class_index_str in [str(i) for i in range(len(CLASSES))]:
                    # Mapear índice a nombre de clase
                    class_name = list(CLASSES.values())[int(class_index_str)]
                    class_counts[class_name] += 1
    
    # --- Bucle Principal ---
    sorted_image_keys = sorted(masks_by_image.keys())
    for img_basename in tqdm(sorted_image_keys, desc="Procesando Imágenes"):
        final_label_path = os.path.join(labels_output_dir, f"{img_basename}.txt")
        if os.path.exists(final_label_path):
            continue

        found_image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(ALL_IMAGES_DIR, f"{img_basename}{ext}")
            if os.path.exists(potential_path):
                found_image_path = potential_path
                break
        
        if not found_image_path: continue
        
        image_bgr = cv2.imread(found_image_path)
        if image_bgr is None: continue

        yolo_lines_for_this_image = []
        
        for i, mask_path in enumerate(masks_by_image[img_basename]):
            # Limpiar la pantalla y mostrar el estado actual
            os.system('cls' if os.name == 'nt' else 'clear')
            print("--- Herramienta de Curación y Clasificación ---")
            print("\n--- CONTEO ACTUAL DE INSTANCIAS ---")
            total_count = 0
            for class_name, count in sorted(class_counts.items()):
                print(f"  - {class_name.capitalize()}: {count}")
                total_count += count
            print(f"  - TOTAL: {total_count}")
            print("------------------------------------")
            
            print("\nInstrucciones de Clasificación:")
            for key, value in CLASSES.items():
                print(f"  Presiona '{key}' para clasificar como: {value.capitalize()}")
            print("  Presiona 'n' para RECHAZAR/IGNORAR.")
            print("  Presiona 'q' para GUARDAR y SALIR.")
            print(f"\nRevisando: {img_basename} (Instancia {i+1}/{len(masks_by_image[img_basename])})")


            candidate_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if candidate_mask is None: continue

            contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) < 100: continue

            display_img = image_bgr.copy()
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 3)

            h_disp, w_disp = display_img.shape[:2]
            if h_disp > 900:
                display_img = cv2.resize(display_img, (int(w_disp * 900 / h_disp), 900))

            window_title = f"{img_basename} - Instancia {i+1}/{len(masks_by_image[img_basename])}"
            cv2.imshow(window_title, display_img)
            
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            key_char = chr(key & 0xFF).lower()

            if key_char in CLASSES:
                class_name = CLASSES[key_char]
                class_index = list(CLASSES.keys()).index(key_char)
                class_counts[class_name] += 1 # Actualizar contador

                polygon_points = contour.flatten().tolist()
                h_orig, w_orig = image_bgr.shape[:2]
                normalized_points = [f"{pt / w_orig if i % 2 == 0 else pt / h_orig:.6f}" for i, pt in enumerate(polygon_points)]
                
                yolo_line = f"{class_index} " + " ".join(map(str, normalized_points))
                yolo_lines_for_this_image.append(yolo_line)

            elif key_char == 'n':
                pass # No hacer nada
            elif key_char == 'q':
                if yolo_lines_for_this_image:
                     shutil.copy(found_image_path, os.path.join(images_output_dir, os.path.basename(found_image_path)))
                     with open(final_label_path, 'w') as f: f.write("\n".join(yolo_lines_for_this_image))
                print("Progreso guardado. ¡Hasta la próxima!")
                return

        if yolo_lines_for_this_image:
            shutil.copy(found_image_path, os.path.join(images_output_dir, os.path.basename(found_image_path)))
            with open(final_label_path, 'w') as f:
                f.write("\n".join(yolo_lines_for_this_image))

    print("\n--- Proceso de curación y clasificación finalizado. ---")

if __name__ == '__main__':
    curate_and_classify()