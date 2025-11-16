import os
import requests
import cv2
from ultralytics import YOLO

# 1️ Download the video if not exists
video_path = "strreet.mp4"
if not os.path.exists(video_path):
    url = "https://github.com/Oogechukwu/datasets/raw/main/strreet.mp4"
    print("Downloading video...")
    r = requests.get(url, stream=True)
    with open(video_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# 2️ Load YOLOv8 model
model = YOLO("yolov8n.pt")  # small pre-trained model, will download automatically

# 3️ Run video inference with streaming
results = model.predict(source=video_path, stream=True, show=False)

# 4 Display each frame in a window
for r in results:
    frame = r.plot()  # Draw boxes and labels
    cv2.imshow("YOLOv8 Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
