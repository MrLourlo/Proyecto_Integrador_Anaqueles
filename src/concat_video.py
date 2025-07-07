import cv2
import os

# Rutas de entrada
video_files = [
    "video_output/demo_part_0.mp4",
    "video_output/demo_part_1.mp4",
    "video_output/demo_part_2.mp4"
]

# Rango de frames por video
frame_ranges = [
    (0, 29),    # demo_part_0.mp4: frames 0–29
    (30, 59),   # demo_part_1.mp4: frames 30–59
    (60, 89)    # demo_part_2.mp4: frames 60–89
]

# Ruta de salida
output_path = "video_output/video_transiciones.mp4"

# Inicializa video final
fps, width, height = None, None, None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

for i, (video_path, (start_f, end_f)) in enumerate(zip(video_files, frame_ranges)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if end_f >= total_frames:
        print(f"⚠️ El video {video_path} solo tiene {total_frames} frames. Ajustando rango...")
        end_f = total_frames - 1

    # Leer solo los frames del rango deseado
    for frame_idx in range(end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= start_f:
            out.write(frame)

    cap.release()

out.release()
print(f"✅ Video final con transiciones guardado en: {output_path}")
