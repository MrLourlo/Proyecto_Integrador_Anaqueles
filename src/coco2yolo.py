import os
import json
import shutil
import random

# Configuración
dataset_dir = "output_dataset/coco_data"
output_dir = "yolo_dataset"
train_ratio = 0.8

os.makedirs(output_dir, exist_ok=True)
for subfolder in ["images/train", "images/val", "annotations"]:
    os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

# Carga anotaciones COCO
with open(os.path.join(dataset_dir, "coco_annotations.json"), "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Mezclar imágenes y dividir en train/val
random.seed(42)
random.shuffle(images)
split_idx = int(len(images) * train_ratio)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Crear diccionarios para rápido acceso
train_img_ids = set(img["id"] for img in train_images)
val_img_ids = set(img["id"] for img in val_images)

def filter_annotations(img_ids):
    return [ann for ann in annotations if ann["image_id"] in img_ids]

train_anns = filter_annotations(train_img_ids)
val_anns = filter_annotations(val_img_ids)

# Función para copiar imágenes a la carpeta correspondiente
def copy_images(image_list, src_dir, dst_dir):
    for img in image_list:
        src_path = os.path.join(src_dir, img["file_name"])
        dst_path = os.path.join(dst_dir, img["file_name"])
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

# Copiar imágenes
copy_images(train_images, os.path.join(dataset_dir, "imgs"), os.path.join(output_dir, "images/train"))
copy_images(val_images, os.path.join(dataset_dir, "imgs"), os.path.join(output_dir, "images/val"))

# Guardar anotaciones COCO filtradas
def save_coco_json(images, annotations, categories, filename):
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(filename, "w") as f:
        json.dump(coco_format, f)

save_coco_json(train_images, train_anns, categories, os.path.join(output_dir, "annotations/instances_train.json"))
save_coco_json(val_images, val_anns, categories, os.path.join(output_dir, "annotations/instances_val.json"))

# Crear archivo data.yaml
data_yaml = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
"""
for cat in categories:
    data_yaml += f"  {cat['id']}: {cat['name']}\n"

with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("Dataset preparado en:", os.path.abspath(output_dir))
