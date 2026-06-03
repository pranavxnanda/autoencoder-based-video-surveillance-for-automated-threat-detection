import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img  # safer for TF 2.13+

image_data = []

train_path = "Avenue Dataset/training_videos"
fps = 5
train_videos = os.listdir(train_path)
train_images_path = os.path.join(train_path, 'frames')

if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

def data_store(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    image_data.append(gray)

for video in train_videos:
    video_path = os.path.join(train_path, video)
    print(f"Processing video: {video}")

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0

    while success:
        frame_path = os.path.join(train_images_path, f"{count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        count += fps
    cap.release()

    print(f"Extracted {count} frames from {video}")

    images = os.listdir(train_images_path)
    for image in images:
        image_path = os.path.join(train_images_path, image)
        data_store(image_path)

print("Converting to NumPy array...")
image_data = np.array(image_data)
a, b, c = image_data.shape
image_data.resize(b, c, a)
image_data = (image_data - image_data.mean()) / (image_data.std())
image_data = np.clip(image_data, 0, 1)

np.save('training.npy', image_data)
print("Saved training data to training.npy")