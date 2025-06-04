import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
COCO_PATH = 'output_dataset/coco_data/coco_annotations.json'   # Tu archivo COCO completo
IMG_SRC_DIR = 'output_dataset/coco_data/imgs'                   # Carpeta con todas las imágenes
OUT_DIR = 'yolo_dataset'                         # Carpeta destino para YOLOv8
SPLIT_RATIO = 0.8                                # 80% train, 20% val

# === CREAR CARPETAS DESTINO ===
paths = ['images/train', 'images/val', 'labels/train', 'labels/val']
for p in paths:
    Path(f"{OUT_DIR}/{p}").mkdir(parents=True, exist_ok=True)

# === CARGAR ANOTACIONES COCO ===
with open(COCO_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# === DIVIDIR IMÁGENES EN TRAIN/VAL ===
random.shuffle(images)
split_index = int(len(images) * SPLIT_RATIO)
train_images = images[:split_index]
val_images = images[split_index:]

image_id_to_split = {
    img['id']: 'train' if img in train_images else 'val' for img in images
}

# === AGRUPAR ANOTACIONES POR IMAGEN ===
ann_map = {}
for ann in annotations:
    if ann['image_id'] not in ann_map:
        ann_map[ann['image_id']] = []
    ann_map[ann['image_id']].append(ann)

# === FUNCIONES DE CONVERSIÓN ===
def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    return [
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h
    ]

def convert_and_save(image, split):
    image_id = image['id']
    file_name = image['file_name']
    w, h = image['width'], image['height']
    img_src = os.path.join(IMG_SRC_DIR, file_name)
    img_dst = os.path.join(OUT_DIR, f'images/{split}/{file_name}')
    shutil.copyfile(img_src, img_dst)

    label_path = os.path.join(OUT_DIR, f'labels/{split}/{Path(file_name).with_suffix(".txt")}')
    with open(label_path, 'w') as f:
        for ann in ann_map.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue  # ignorar anotaciones con máscara tipo RLE
            if not ann.get("segmentation") or len(ann["segmentation"]) == 0:
                continue  # ignorar si no hay segmentación válida

            cat_id = ann['class_id']
            bbox = coco_to_yolo_bbox(ann['bbox'], w, h)
            segm = ann['segmentation'][0]  # solo el primer polígono
            norm_segm = [str(coord / w if i % 2 == 0 else coord / h) for i, coord in enumerate(segm)]
            line = f"{cat_id} {' '.join(map(str, bbox))} {' '.join(norm_segm)}\n"
            f.write(line)


print("🔄 Convirtiendo dataset COCO → YOLOv8 formato segmentación...")
for img in tqdm(images):
    split = image_id_to_split[img['id']]
    convert_and_save(img, split)

# === GENERAR data.yaml ===
data_yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(data_yaml_path, "w") as f:
    f.write(f"path: {OUT_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for cat_id in sorted(categories.keys()):
        f.write(f"  {cat_id}: {categories[cat_id]}\n")

print(f"✅ Conversión completa. Archivo data.yaml generado en {data_yaml_path}")
