from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open video file (change path to your video)
video_path = 'videos/test2.mp4'  
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: Save output video
out = cv2.VideoWriter('output_detected.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      fps, 
                      (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break

    # Object detection
    results = model(frame)
    
    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Display frame
    cv2.imshow("Video Object Detection", annotated_frame)
    
    # Save frame to output video
    out.write(annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()