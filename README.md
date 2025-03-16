# Real-Time-Human-Activity-Recognition
A computer vision project built to recognize human actions in real time. <br />

## User Manual:

1. **Install Dependencies**  
   Install the required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt

2. **Model Files**<br />
   The required model files (resnet-34_kinetics.onnx, yolov8s.pt, Actions.txt) are already included in the GitHub repository. You do not need to download them separately. Just      make sure these files are in the same directory as the script when running the program.

3. **CUDA, MPS, or CPU?**<br />
   If your machine has an Nvidia GPU with CUDA, you should enable it for better performance.
   If you are using Apple Silicon (M1, M2, etc.), select MPS.
   Otherwise, CPU will be used by default.

4. **Input Options**<br />
   The program supports both live webcam feed or a pre-recorded video file as input. Select the desired input type when prompted.

5. **Model Customization**<br />
   You can choose between different YOLO models (s, m, l, x). Higher models provide more accurate object detection but may introduce latency in real-time use.

6. **Running the Program**<br />
   Both HAR Resnet.py and HAR multi-person.py can be run independently.
   HAR Resnet.py processes the entire frame for activity recognition.
   HAR multi-person.py processes frames by detecting multiple humans and recognizing their actions individually.

7. **Save Video Option**<br />
   Optionally, you can save the output video by selecting "y" when prompted. Provide a path to save the video.

8. **Exit Controls**<br />
   To stop the program, press Esc or Space to exit the loop gracefully.

## Packages Used:

* **OpenCV**: A library used for image processing and computer vision tasks, such as breaking down videos into individual frames and manipulating those frames (e.g., resizing, displaying, saving).<br />
* **Ultralytics**: A library for working with YOLO models, responsible for loading and running the YOLOv8 model for object detection.<br />
* **Numpy**: A library used for numerical and array operations, including mathematical calculations and managing image data arrays.<br />

## Models Used:

* **resnet-34_kinetics.onnx**: A pre-trained CNN model with 34 layers, used for human action recognition. The model is trained on the Kinetics-400 dataset, which consists of 400 different human action classes. This model helps in identifying what activity a person is performing in a video.<br />
* **yolov8s.pt**: A pre-trained YOLOv8 model used for object detection, specifically to detect humans (class ID 0) in a frame. This model identifies and localizes people in the video, allowing for individual action recognition for each detected person.<br />

## How It Works:

This project processes videos (either from a live webcam feed or a pre-recorded file) to recognize human activities in real-time. The main steps are:

1. **Video Breakdown**:<br />
   The video is broken down into individual frames. Each frame is processed by the system to detect human actions.<br />
2. **Human Detection**:<br />
   The YOLOv8 model detects humans in the video frame. It identifies the bounding boxes around each person in the frame. If multiple people are present, each person is treated      as a separate "track."<br />
3. **Activity Recognition**:<br />
   After detecting the humans, the system extracts the region of interest (ROI) for each person (i.e., the bounding box around them). The activity recognition model (ResNet-34)     then classifies the activity based on these cropped frames.<br />
4. **SAMPLE_DURATION and SAMPLE_SIZE**:<br />
   * SAMPLE_DURATION (16) is the number of consecutive frames from the video that are grouped together to recognize a single action. It ensures that the system looks at a             sequence of frames rather than just one to understand the action over time. For example, if a person is walking, it requires a few frames to capture the movement before          making a decision on the activity.<br />
   * SAMPLE_SIZE (112) is the size to which each cropped human region (ROI) is resized. This ensures consistency in the input to the activity recognition model, as the model          expects a fixed size for each frame (112x112 pixels). All the human regions, regardless of their original size in the video, are resized to this size for accurate                recognition <br />
5. **Labeling the Activities**:<br />
   After processing the frames, the recognized activity label (e.g., "walking," "running") is displayed on the video feed. If multiple people are detected, each one has a label     showing their individual activity.<br />
6. **Saving the Video (optional)**:<br />
   If you choose to save the processed video, the system writes the labeled frames into a new video file, allowing you to review the results later.<br />
