import cv2
import time
from ultralytics import YOLO

# preload models
model_n = YOLO("D:/Downloads/yolov8n-trash (1)/kaggle/working/runs/detect/trash-detection4/weights/last.pt")
model_l = YOLO("D:/Projects/Capstone Project/trained model weight/V8n.pt")

# open same video
cap = cv2.VideoCapture("D:/Downloads/archive (7)/trash_icra_xml/inference_data/several.mp4")

# store stats
fps_n, fps_l = 0, 0
frames_n, frames_l = 0, 0
time_n, time_l = 0, 0
 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # clone frame so both models get the same input
    frame_n = frame.copy()
    frame_l = frame.copy()

    # --- YOLOv8n ---
    start = time.time()
    results_n = model_n(frame_n, conf=0.5, verbose=False)
    end = time.time()
    time_n += (end - start)
    frames_n += 1
    out_n = results_n[0].plot()

    # --- YOLOv8l ---
    start = time.time()
    results_l = model_l(frame_l, conf=0.5, verbose=False)
    end = time.time()
    time_l += (end - start)
    frames_l += 1
    out_l = results_l[0].plot()

    # resize side by side display
    h, w = frame.shape[:2]
    out_n = cv2.resize(out_n, (w//2, h//2))
    out_l = cv2.resize(out_l, (w//2, h//2))
    combined = cv2.hconcat([out_n, out_l])

    cv2.imshow("YOLOv8n (left) vs YOLOv8l (right)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# final stats
fps_n = frames_n / time_n if time_n > 0 else 0
fps_l = frames_l / time_l if time_l > 0 else 0

print(f"YOLOv8n average FPS: {fps_n:.2f}")
print(f"YOLOv8l average FPS: {fps_l:.2f}")
