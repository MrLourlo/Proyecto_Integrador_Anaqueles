# generate_pseudo_labels.py
import os
import glob
from ultralytics import YOLO
import torch
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# 1. Ruta al MEJOR modelo que hemos entrenado
MODEL_PATH = 'runs/segment/mantecadas_multiclass_teacher_run12/weights/best.pt' # Ajusta el nombre de la corrida si es necesario

# 2. Ruta a la carpeta con TODAS las imágenes (las 3,478)
FULL_IMAGES_DIR = r'C:\Users\Brandon\Documents\MNA\Bimbo\Bimbo\Fotos Chambita 1364' # <--- CAMBIA ESTO

# 3. Carpeta de salida para las etiquetas generadas por la IA
PSEUDO_LABELS_DIR = 'pseudo_labels'

# 4. Umbral de confianza para guardar una predicción
CONFIDENCE_THRESHOLD = 0.25 # Empezamos con un umbral bajo para capturar todo lo posible
# --- FIN DE LA CONFIGURACIÓN ---

def generate_labels():
    """
    Usa el modelo entrenado para predecir etiquetas en un gran conjunto de datos.
    """
    print("--- FASE 1: Generando Pseudo-Etiquetas con el Modelo Guía ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    os.makedirs(PSEUDO_LABELS_DIR, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(FULL_IMAGES_DIR, '*.[jJ][pP]*[gG]')) + \
                  glob.glob(os.path.join(FULL_IMAGES_DIR, '*.[pP][nN][gG]'))
    
    print(f"Se procesarán {len(image_paths)} imágenes.")

    for img_path in tqdm(image_paths, desc="Generando etiquetas"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(PSEUDO_LABELS_DIR, f"{base_name}.txt")

        # Ejecutar predicción
        results = model(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

        yolo_lines = []
        for r in results:
            if r.masks is None:
                continue
            
            h, w = r.orig_shape
            for mask, box in zip(r.masks, r.boxes):
                class_index = int(box.cls.item())
                
                # Convertir la máscara a polígono y normalizar
                polygon_points = mask.xy[0].flatten().tolist()
                normalized_points = [f"{pt / w if i % 2 == 0 else pt / h:.6f}" for i, pt in enumerate(polygon_points)]
                
                line = f"{class_index} " + " ".join(map(str, normalized_points))
                yolo_lines.append(line)
        
        # Guardar el archivo de etiqueta si se encontró algo
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                
    print(f"\n--- Proceso finalizado. {len(os.listdir(PSEUDO_LABELS_DIR))} archivos de etiquetas generados en la carpeta '{PSEUDO_LABELS_DIR}'. ---")

if __name__ == '__main__':
    generate_labels()