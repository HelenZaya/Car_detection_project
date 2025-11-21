from ultralytics import YOLO
import cv2

def main():
    model = YOLO(r"C:\Users\wuttudbe\OneDrive\Documents\GitHub\Car_detection_project\runs\detect\train13\weights\best.pt")

    video_path = "videos/test2.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        cv2.imshow("TYPE DETECTION", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()