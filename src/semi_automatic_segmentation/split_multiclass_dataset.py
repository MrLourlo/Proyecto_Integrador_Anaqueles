# split_multiclass_dataset.py
import os
import glob
import random
import shutil
from tqdm import tqdm

# --- CONFIGURACIÓN (AJUSTADA PARA LA FASE FINAL) ---
# La carpeta de entrada es nuestro dataset masivo y verificado
INPUT_DATASET_DIR = 'final_verified_dataset' 

# La carpeta de salida para el entrenamiento definitivo
YOLO_FINAL_DIR = 'mantecadas_yolo_definitivo' # <-- Nuevo nombre para la salida final

VALIDATION_SPLIT = 0.2
# --- FIN DE LA CONFIGURACIÓN ---

def split_data():
    print(f"--- Dividiendo el Dataset Definitivo desde '{INPUT_DATASET_DIR}' ---")
    images_source_dir = os.path.join(INPUT_DATASET_DIR, 'images')
    labels_source_dir = os.path.join(INPUT_DATASET_DIR, 'labels')

    # ... el resto del script es idéntico y no necesita cambios ...
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_FINAL_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(YOLO_FINAL_DIR, 'labels', subset), exist_ok=True)

    image_filenames = sorted(os.listdir(images_source_dir))
    random.seed(42)
    random.shuffle(image_filenames)

    split_point = int(len(image_filenames) * (1 - VALIDATION_SPLIT))
    train_files, val_files = image_filenames[:split_point], image_filenames[split_point:]

    print(f"Dataset total a usar: {len(image_filenames)} imágenes. Train: {len(train_files)}, Val: {len(val_files)}.")

    def copy_files(file_list, subset):
        for filename in tqdm(file_list, desc=f"Copiando {subset}"):
            basename = os.path.splitext(filename)[0]
            shutil.copy(os.path.join(images_source_dir, filename), os.path.join(YOLO_FINAL_DIR, 'images', subset, filename))
            shutil.copy(os.path.join(labels_source_dir, f"{basename}.txt"), os.path.join(YOLO_FINAL_DIR, 'labels', subset, f"{basename}.txt"))

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    print(f"\n--- División finalizada. Dataset definitivo listo en: '{YOLO_FINAL_DIR}' ---")

if __name__ == '__main__':
    split_data()