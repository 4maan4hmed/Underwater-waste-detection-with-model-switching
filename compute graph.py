import cv2, time, psutil, csv
from ultralytics import YOLO
import pynvml

# init GPU monitoring
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

# preload models
model_n = YOLO("D:/Projects/Capstone Project/trained model weight/V8n.pt")
model_l = YOLO("D:/Projects/Capstone Project/trained model weight/V8l.pt")

cap = cv2.VideoCapture("D:/Projects/Capstone Project/image/waste.mp4")

target_fps = 2
frame_interval = 1.0 / target_fps

# CSV setup
csv_file = open("yolo_resource_usage.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "Frame", 
    "Model", 
    "CPU(%)", 
    "RAM(%)", 
    "GPU(%)", 
    "VRAM(MB)", 
    "Power(W)"
])

def log_resources(frame_id, model_name):
    cpu = psutil.cpu_percent(interval=0.1)  # fixed
    ram = psutil.virtual_memory().percent
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # MB

    writer.writerow([frame_id, model_name, cpu, ram, util.gpu, mem, power])
    print(f"Logged {model_name} at frame {frame_id}")  # progress output


frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    start = time.time()

    # --- YOLOv8n ---
    _ = model_n(frame, conf=0.5, verbose=False)
    log_resources(frame_id, "YOLOv8n")

    # --- YOLOv8l ---
    _ = model_l(frame, conf=0.5, verbose=False)
    log_resources(frame_id, "YOLOv8l")

    # enforce same FPS
    elapsed = time.time() - start
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
pynvml.nvmlShutdown()
