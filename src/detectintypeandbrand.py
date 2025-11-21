from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # Load models
    type_model = YOLO(r"C:\Users\wuttudbe\OneDrive\Documents\GitHub\Car_detection_project\runs\detect\train5\weights\best.pt")
    brand_model = YOLO(r"C:\Users\wuttudbe\OneDrive\Documents\GitHub\Car_detection_project\runs\detect\train13\weights\best.pt")

    # Open video
    video_path = 'videos/test2.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    out = cv2.VideoWriter(
        'output_type_brand_colored.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished!")
            break

        # Optional: resize for faster inference
        input_frame = cv2.resize(frame, (640, 640))

        # Run models
        type_results = type_model(input_frame)[0]
        brand_results = brand_model(input_frame)[0]

        # Scale back boxes to original frame size
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 640

        # Draw car type boxes in blue
        for box in type_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            xyxy = (int(xyxy[0]*scale_x), int(xyxy[1]*scale_y),
                    int(xyxy[2]*scale_x), int(xyxy[3]*scale_y))
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Type {cls} {conf:.2f}", (xyxy[0], xyxy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw car brand boxes in green
        for box in brand_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            xyxy = (int(xyxy[0]*scale_x), int(xyxy[1]*scale_y),
                    int(xyxy[2]*scale_x), int(xyxy[3]*scale_y))
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Brand {cls} {conf:.2f}", (xyxy[0], xyxy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show and write frame
        cv2.imshow("Car Type + Brand Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
