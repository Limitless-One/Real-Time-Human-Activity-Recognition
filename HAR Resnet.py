import numpy as np
import imutils
import cv2

ACT = open("Actions.txt").read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

print("Real Time Human Activity Recognition using Resnet:")
gp = cv2.dnn.readNet("resnet-34_kinetics.onnx")

# Check if GPU is to be used
gpu = input("Would you like to use your GPU? [y/n]: ")
if gpu == "y":
    #setting preferable backend and target to CUDA
    gp.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    gp.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Select input source (live feed or uploaded video)
feed = input("Would you like to use live feed (webcam)? [y/n]: ")
if feed == "y":
    vs = cv2.VideoCapture(0)
else:
    path = input("Enter video path: ")
    vs = cv2.VideoCapture(path)

# Get FPS of the video feed (either live feed or uploaded video)
fps = vs.get(cv2.CAP_PROP_FPS)
print("Original FPS: ", fps)

# Ask if user wants to save the video
save = input("Would you like to save the video? [y/n]: ")
writer = None

# Initialize video writer only if saving the video
if save == "y":
    save_path = input("Enter video path (e.g., output.mp4):")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for better compatibility
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height), True)

# Main processing loop
while True:
    frames = []  # frames for processing
    originals = []  # original frames

    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()

        if not grabbed:
            print("[INFO] End of video reached or failed to read frame. Stopping...")
            break  # Exit the loop if we reach the end of the video or fail to read a frame

        originals.append(frame)
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    if not grabbed:  # If end of video is reached, break the outer loop
        break

    # Blob construction and activity recognition
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    gp.setInput(blob)
    outputs = gp.forward()
    label = ACT[np.argmax(outputs)]

    # Add labels to frames
    for frame in originals:
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Activity Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # Break if Esc or Space key is pressed
        if key == 27 or key == 32:
            print("[INFO] Stopping...")
            break

        # Save the frame to video
        if writer is not None:
            writer.write(frame)

    if key == 27 or key == 32:
        break

# Release resources
if writer is not None:
    writer.release()

if vs is not None:
    vs.release()

cv2.destroyAllWindows()  # Close the OpenCV display window
