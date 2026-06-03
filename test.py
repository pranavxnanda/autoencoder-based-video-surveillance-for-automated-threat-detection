import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import imutils

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance

# Load trained model
model_path = os.path.join("model", "saved_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = load_model(model_path)

# Define video path
video_path = os.path.join("Avenue Dataset", "testing_videos", "05.avi")

# Check if the video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found at: {video_path}")

# Open video capture
cap = cv2.VideoCapture(video_path)
print("Video Opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError(f"Unable to open video file: {video_path}")

while True:
    im_frames = []
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Collect 10 consecutive frames
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        image = imutils.resize(frame, width=700, height=600)

        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

    if len(im_frames) < 10:
        print("Not enough frames left to form a sequence.")
        break

    im_frames = np.array(im_frames)
    im_frames.resize(227, 227, 10)
    im_frames = np.expand_dims(im_frames, axis=0)
    im_frames = np.expand_dims(im_frames, axis=4)

    output = model.predict(im_frames)
    loss = mean_squared_loss(im_frames, output)
    print("Mean Squared Loss:", loss)

    # Abnormal event detection
    if 0.000387 < loss:
        print('Abnormal Event Detected')
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)

        text = "Abnormal Event"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Draw a white background for text
        cv2.rectangle(image, (50, 50 - text_height), (50 + text_width, 50), (255, 255, 255), -1)
        cv2.putText(image, text, (45, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    resized_frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Video Surveillance for Cyber Threat Detection", resized_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("User interrupted — exiting.")
        break

cap.release()
cv2.destroyAllWindows()