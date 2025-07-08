from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        print(f"CUDA disponible. Usando GPU: {torch.cuda.get_device_name(0)}")

    # Mantenemos el modelo mediano, que es un excelente balance de potencia y velocidad.
    model = YOLO('yolov8m-seg.pt')

    print("Iniciando entrenamiento del MODELO DEFINITIVO...")
    results = model.train(
        data='mantecadas_multiclass.yaml', # Apunta al mismo YAML
        epochs=200,                       # <-- Aumentamos las epochs para que tenga más tiempo de aprender del gran dataset
        imgsz=640,
        batch=8,                          # Mantenemos el batch en 8, es un valor seguro
        workers=0,
        augment=True,                     # La aumentación sigue siendo clave
        patience=50,                      # <-- Nuevo: le decimos que pare si no mejora en 50 epochs
        name='mantecadas_DEFINITIVO_run1' # <-- El nombre de nuestro campeón
    )
    print("¡Entrenamiento definitivo finalizado!")

if __name__ == '__main__':
    main()