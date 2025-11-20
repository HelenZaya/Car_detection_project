from ultralytics import YOLO

# 1) Load pre-trained model
model = YOLO("yolov8n.pt")  # or yolov8s.pt etc.

# 2) Train
model.train(
    data="./data/carbrand_dataset/carbrand.yaml",  # path to data.yaml
    epochs=30,
    imgsz=3226,
    batch=16,     # adjust if GPU memory is small
    workers=2,    # reduce on Windows if DataLoader error
)