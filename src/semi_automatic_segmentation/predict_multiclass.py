# predict_multiclass.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# --- CONFIGURACIÓN ---
# 1. Ruta a tu NUEVO y MEJOR modelo entrenado
MODEL_PATH = r'C:\Users\Brandon\Documents\MNA\Bimbo\runs\segment\mantecadas_DEFINITIVO_run1\weights\best.pt' # Asegúrate que el nombre de la carpeta sea el correcto

# 2. Ruta a la imagen que quieres probar
IMAGE_PATH = r'C:\Users\Brandon\Documents\MNA\Bimbo\image.png' # <--- CAMBIA ESTO

# 3. Umbral de confianza
CONFIDENCE_THRESHOLD = 0.2 # Podemos ser más exigentes ahora, 50% de confianza
# --- FIN DE LA CONFIGURACIÓN ---

def run_multiclass_prediction():
    """
    Carga el modelo multi-clase, ejecuta la predicción y muestra los resultados con colores por clase.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    try:
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    try:
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {IMAGE_PATH}")
    except Exception as e:
        print(e)
        return
        
    print("Ejecutando predicción con el modelo Multi-Clase...")
    results = model(img)

    output_img = img.copy()
    overlay = output_img.copy()

    # --- NUEVO: Colores para cada clase ---
    # BGR (Azul, Verde, Rojo)
    class_colors = [
        (255, 0, 0),    # original = Azul
        (0, 255, 0),    # nuez = Verde
        (0, 0, 255),    # marmoleada = Rojo
        (0, 255, 255),  # chocolate = Amarillo
    ]

    for result in results:
        if result.masks is None:
            print("No se encontraron objetos.")
            continue

        for mask, box in zip(result.masks, result.boxes):
            if box.conf.item() > CONFIDENCE_THRESHOLD:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                confidence = box.conf.item()
                color = class_colors[class_id]

                # Dibujar máscara semitransparente
                polygon_points = mask.xy[0].astype(int)
                cv2.fillPoly(overlay, [polygon_points], color, lineType=cv2.LINE_AA)

                # Dibujar caja y etiqueta
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(output_img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)
    
    output_filename = 'prediction_multiclass_result.jpg'
    cv2.imwrite(output_filename, output_img)
    print(f"Imagen con resultados guardada como: {output_filename}")

    cv2.imshow('Resultado - Modelo Multi-Clase', output_img)
    print("Presiona cualquier tecla para salir.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_multiclass_prediction()