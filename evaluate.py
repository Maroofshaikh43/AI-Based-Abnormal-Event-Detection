import time
import torch
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import os
from pathlib import Path
import yaml

# Configuration
MODEL_PATH = "D:/Final_Project/project/Abnormal-Event-Detection/runs/combined_incidents/yolov8_incident_detector2/weights/best.pt"
DATASET_YAML = "D:/Final_Project/project/Abnormal-Event-Detection/datasets/combined_incidents/combined_incidents.yaml"
IMG_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data['names']

def evaluate_model():
    model = YOLO(MODEL_PATH)
    class_names = load_class_names(DATASET_YAML)

    results = model.val(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )

    metrics = results.box  # This is a Metric object

    for i, name in enumerate(class_names):
        p, r, ap50, ap = metrics.class_result(i)
        print(f"{name:<18s} | P: {p:.3f} | R: {r:.3f} | AP50: {ap50:.3f} | AP: {ap:.3f}")
    # Correct way to access mean metrics
    mp, mr, map50, map = metrics.mean_results()

    print(f"\n--- Evaluation Metrics ---")
    print(f"Precision: {mp:.4f}")
    print(f"Recall: {mr:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map:.4f}")



def measure_inference_speed():
    model = YOLO(MODEL_PATH)
    test_image = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(5):
        model.predict(source=test_image, imgsz=IMG_SIZE, device="cuda", verbose=False)

    # Measure time
    start = time.time()
    N = 100
    for _ in range(N):
        model.predict(source=test_image, imgsz=IMG_SIZE, device="cuda", verbose=False)
    end = time.time()
    avg_time = (end - start) / N
    print(f"\nAverage Inference Time per Frame: {avg_time*1000:.2f} ms ({1/avg_time:.2f} FPS)")

if __name__ == "__main__":
    evaluate_model()
    measure_inference_speed()
