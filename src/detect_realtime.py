from ultralytics import YOLO
import cv2

def main():
    # Load your trained models
    type_model = YOLO(r"C:\Users\wuttudbe\OneDrive\Documents\GitHub\Car_detection_project\runs\detect\train13\weights\best.pt")
    brand_model = YOLO(r"C:\Users\wuttudbe\OneDrive\Documents\GitHub\Car_detection_project\runs\detect\train5\weights\best.pt")

    # Open video
    video_path = 'videos/test2.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    out = cv2.VideoWriter(
        'output_type_brand.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished!")
            break

        # Run detection on both models
        type_results = type_model(frame)[0]
        brand_results = brand_model(frame)[0]

        # Plot results safely
        annotated_frame = type_results.plot()
        # Only overlay brand results if plot() supports passing an image
        for box in brand_results.boxes:
            # Extract coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Draw rectangle
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Brand {cls} {conf:.2f}", (xyxy[0], xyxy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show annotated frame
        cv2.imshow("Car Type + Car Brand Detection", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
