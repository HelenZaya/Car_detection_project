from ultralytics import YOLO

# 1) Load pre-trained model
model = YOLO("yolov8n.pt")  # or yolov8s.pt etc.

# 2) Train
model.train(
    data="cardata_data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,     # adjust if GPU memory is small
    workers=2,    # reduce on Windows if DataLoader error
)