from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    result = model.train(
        data="datasets/combined_incidents/combined_incidents.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0,  # Prevents multiprocessing issues on Windows
        project="runs/combined_incidents",
        name="yolov8_incident_detector",
        device="cuda"
    )
