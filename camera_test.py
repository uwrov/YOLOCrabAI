from typing import Any
import cv2
from ultralytics import YOLO

# Settings
video_file = "/Users/chasecarson/Desktop/yolo_crabs/YOLOCrabAI/TestVideos/IMG_1173.MOV"  # Local video file
#stream_url = "" 
#camera_id = 0  # Camera index (0 = default webcam, 1, 2, etc. for other cameras)

source = video_file  # Change to camera_id for camera, stream_url for stream

conf_threshold = 0.8
iou_threshold = 0.3 # NMS IoU threshold (lower = more aggressive suppression)
egc_class_id = 1

# Load model
model = YOLO("/Users/ChaseCarson/Desktop/yolo_crabs/YOLOCrabAI/Crabs.v3i.yolov11/runs/detect/train2/weights/best.pt")

print("Model loaded! Press 'q' to quit.")

# Counting variables
total_crabs = 0
seen_ids = set()  # All track IDs we've ever seen in this video

# Main loop with tracking
for result in model.track(source=source, stream=True, tracker="bytetrack.yaml", persist=True, verbose=False, iou=iou_threshold):
    frame = result.orig_img
    boxes = result.boxes
    current_ids = set()

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None

        if cls == egc_class_id and conf >= conf_threshold:
            if track_id is not None:
                current_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"egc id={track_id} {conf:.2f}" if track_id else f"egc {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Update counter - count each unique track ID only once
    for track_id in current_ids:
        if track_id not in seen_ids:
            seen_ids.add(track_id)
            total_crabs += 1

    cv2.putText(frame, f"Total crabs: {total_crabs}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ROV EGC Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cv2.destroyAllWindows()
