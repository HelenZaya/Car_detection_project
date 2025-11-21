from ultralytics import YOLO
import cv2

def main():
    model = YOLO(r"D:\VS studio amka\object-detection-project\Car_detection_project\runs\detect\train5\weights\best.pt")

    video_path = "videos/test2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    out = cv2.VideoWriter(
        'output_cartype_detected.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Video finished! Processed {frame_count} frames.")
            break

        results = model(frame)[0]
        annotated = results.plot()
        
        # Write annotated frame to output video
        out.write(annotated)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Output saved to: output_cartype_detected.mp4")

if __name__ == "__main__":
    main()