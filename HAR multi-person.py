import cv2
import numpy as np
import imutils
from ultralytics import YOLO

# ----------------------- Parameters and Model Setup -----------------------
# Load activity labels
ACT = open("Actions.txt").read().strip().split("\n")
SAMPLE_DURATION = 16  # number of frames per clip
SAMPLE_SIZE = 112  # size to which each cropped human ROI is resized

# Load the ResNet activity recognition model (ONNX)
activity_net = cv2.dnn.readNet("resnet-34_kinetics.onnx")
use_gpu = input("Would you like to use GPU for activity recognition? [y/n]: ").strip().lower() == "y"
if use_gpu:
    activity_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    activity_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

yolo_device = "cuda" if use_gpu else "cpu"
if not use_gpu:
    use_mps = input("Would you like to use MPS instead? [y/n]: ").strip().lower() == "y"
    yolo_device = "mps" if use_mps else "cpu"

# Load YOLOv8 model for human detection
print("Loading YOLOv8 model for human detection...")
yolo_model = YOLO('yolov8s.pt')


# ----------------------- Video Setup -----------------------
# Choose video source
feed_choice = input("Use live feed (webcam)? [y/n]: ").strip().lower() == "y"
if feed_choice:
    cap = cv2.VideoCapture(0)
else:
    video_path = input("Enter video path: ").strip()
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Get video properties for writing output if needed
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}")

save_output = input("Would you like to save the output video? [y/n]: ").strip().lower() == "y"
writer = None
if save_output:
    output_path = input("Enter output video path (e.g., output.mp4): ").strip()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ----------------------- Simple Tracking Setup -----------------------
# Each track corresponds to a detected human. For simplicity, we use a minimal tracker:
tracks = []  # list of dictionaries, each storing: id, current bbox, center, a clip buffer, label, and lost frame count
next_track_id = 0
DIST_THRESHOLD = 50  # maximum distance (pixels) for matching detection to an existing track
MAX_LOST = 10  # remove track if not detected for these many frames


def get_center(bbox):
    x, y, x2, y2 = bbox
    return ((x + x2) // 2, (y + y2) // 2)


# ----------------------- Main Processing Loop -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame grabbed; ending video processing.")
        break

    # Run human detection with YOLOv8
    results = yolo_model(frame, device=yolo_device)
    result = results[0]
    try:
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
        classes = np.array(result.boxes.cls.cpu(), dtype=int)
    except Exception:
        bboxes = np.array(result.boxes.xyxy, dtype=int)
        classes = np.array(result.boxes.cls, dtype=int)

    detections = []
    # Only consider detections with class 0 ("person")
    for cls, bbox in zip(classes, bboxes):
        if cls == 0:
            detections.append(bbox)

    # Keep track of which detection is already assigned to a track
    assigned = [False] * len(detections)

    # Update existing tracks by matching detections (using center distance)
    for track in tracks:
        updated = False
        for i, det_bbox in enumerate(detections):
            if assigned[i]:
                continue
            det_center = get_center(det_bbox)
            track_center = track['center']
            distance = np.linalg.norm(np.array(det_center) - np.array(track_center))
            if distance < DIST_THRESHOLD:
                # Update track info
                track['bbox'] = det_bbox
                track['center'] = det_center
                x, y, x2, y2 = det_bbox
                # Ensure coordinates are within frame bounds
                x, y = max(0, x), max(0, y)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                roi = frame[y:y2, x:x2]
                # Resize the cropped human region to the required input size
                roi_resized = cv2.resize(roi, (SAMPLE_SIZE, SAMPLE_SIZE))
                track['buffer'].append(roi_resized)
                # Keep the buffer size at SAMPLE_DURATION (sliding window)
                if len(track['buffer']) > SAMPLE_DURATION:
                    track['buffer'].pop(0)
                track['lost'] = 0
                updated = True
                assigned[i] = True
                break
        if not updated:
            track['lost'] += 1

    # Create new tracks for unmatched detections
    for i, det_bbox in enumerate(detections):
        if not assigned[i]:
            det_center = get_center(det_bbox)
            x, y, x2, y2 = det_bbox
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            roi = frame[y:y2, x:x2]
            roi_resized = cv2.resize(roi, (SAMPLE_SIZE, SAMPLE_SIZE))
            new_track = {
                'id': next_track_id,
                'bbox': det_bbox,
                'center': det_center,
                'buffer': [roi_resized],
                'label': "",
                'lost': 0
            }
            next_track_id += 1
            tracks.append(new_track)

    # Remove tracks that have not been updated for too long
    tracks = [t for t in tracks if t['lost'] <= MAX_LOST]

    # For each track that has a full clip, run activity recognition
    for track in tracks:
        if len(track['buffer']) == SAMPLE_DURATION:
            clip = track['buffer']
            blob = cv2.dnn.blobFromImages(clip, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
                                          (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            activity_net.setInput(blob)
            outputs = activity_net.forward()
            predicted_label = ACT[np.argmax(outputs)]
            track['label'] = predicted_label
            # (Optionally, you could remove the oldest frame to shift the window)
            # track['buffer'].pop(0)

    # Draw the detection and (if available) the activity label for each track
    for track in tracks:
        x, y, x2, y2 = track['bbox']
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        label_text = f"ID {track['id']}: {track['label']}" if track['label'] else f"ID {track['id']}"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show and optionally save the output frame
    cv2.imshow("Human Detection & Activity Recognition", frame)
    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 32:
        print("Exiting...")
        break

# ----------------------- Cleanup -----------------------
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
