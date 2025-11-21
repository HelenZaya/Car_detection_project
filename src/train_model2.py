from ultralytics import YOLO

# 1) Load pre-trained model
model = YOLO("yolov8n.pt")  # or yolov8s.pt etc.

# 2) Train
model.train(
    data="./data/carbrand_dataset/carbrand.yaml",  # path to data.yaml
    epochs=30,
    imgsz=512,
    batch=8,     # adjust if GPU memory is small
    workers=0,
    device=0    # reduce on Windows if DataLoader error
)