import os
# Suppress TensorFlow info messages (keep warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LambdaCallback
import cv2

# Load preprocessed data
training_data = np.load('training.npy').astype('float32')  # reduce memory usage
frames = training_data.shape[2]
frames = frames - frames % 10  # make divisible by 10
training_data = training_data[:, :, :frames]
training_data = training_data.reshape(-1, 227, 227, 10)
training_data = np.expand_dims(training_data, axis=4)
target_data = training_data.copy()

# Constants
epochs = 5
batch_size = 4  # reduced from 16 to lower memory usage

# Create model
model = Sequential()

model.add(Conv3D(filters=128, kernel_size=(11,11,1), strides=(4,4,1),
                 padding='valid', input_shape=(227,227,10,1), activation='tanh'))

model.add(Conv3D(filters=64, kernel_size=(5,5,1), strides=(2,2,1),
                 padding='valid', activation='tanh'))

model.add(ConvLSTM2D(filters=64, kernel_size=(3,3), strides=1,
                     padding='same', dropout=0.4, recurrent_dropout=0.3, return_sequences=True))

model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), strides=1,
                     padding='same', dropout=0.3, return_sequences=True))

model.add(ConvLSTM2D(filters=64, kernel_size=(3,3), strides=1,
                     padding='same', dropout=0.5, return_sequences=True))

model.add(Conv3DTranspose(filters=128, kernel_size=(5,5,1), strides=(2,2,1),
                          padding='valid', activation='tanh'))

model.add(Conv3DTranspose(filters=1, kernel_size=(11,11,1), strides=(4,4,1),
                          padding='valid', activation='tanh'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Callback to show epoch-level progress
progress_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1} finished. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
)

# Train model with progress output
model.fit(
    training_data,
    target_data,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,  # progress bar per epoch
    callbacks=[progress_callback]
)

# Save the model
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/saved_model.keras")