import cv2
from ultralytics import YOLO
import random


def draw_bounding_boxes_without_id(frame, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidence = results[0].boxes.conf.cpu().numpy()

    for box, clss, conf in zip(boxes, classes, confidence):
        # Generate a random color for each object based on its ID

            random.seed(int(clss) + 8)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
            cv2.putText(
                frame,
                f"{model.model.names[clss]} {conf:.2f}",
                (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (50, 255, 50),
                3,
            )
    return frame


def process_video_with_tracking(model, input_video_path, show_video=True, save_video=False,
                                output_video_path="output_video.mp4"):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=608, verbose=False,
                              tracker="bytetrack.yaml", classes=0)
        results_detect = model_detect.predict(frame, iou=0.4, conf=0.5, imgsz=608, verbose=False)

        
        if results_detect[0].boxes != None:
            draw_bounding_boxes_without_id(frame, results_detect)

        if save_video:
            out.write(frame)

        if show_video:
            frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the input video capture and output video writer
    cap.release()
    if save_video:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    return results_detect, results


# Example usage:
model = YOLO('runs/detect/train6/weights/best.pt')
model_detect = YOLO('runs/detect/train6/weights/best.pt')
model.fuse()
model_detect.fuse()
results_detect, results = process_video_with_tracking(model, "video.MP4", show_video=True, save_video=True,
                                                      output_video_path="output_video.mp4")

