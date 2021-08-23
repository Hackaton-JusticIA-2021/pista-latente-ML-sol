import numpy as np
import cv2
import os
import keras
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from glob import glob
from keras import layers
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib

input_dir_1 = "unet/images/"
target_dir_1 = "unet/target/"
input_dir_2= "data/images/"
target_dir_2 = "data/target/"
img_size = (32, 32)
num_classes = 2
batch_size = 32

input_img_paths_1 = sorted(glob(os.path.join(input_dir_1, '*' + '.png')))
target_img_paths_1 = sorted(glob(os.path.join(target_dir_1, '*' + '.png')))
input_img_paths_2 = sorted(glob(os.path.join(input_dir_2, '*' + '.png')))
target_img_paths_2 = sorted(glob(os.path.join(target_dir_2, '*' + '.png')))

input_img_paths = input_img_paths_1 + input_img_paths_2
target_img_paths = target_img_paths_1 + target_img_paths_2

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

class Patches(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.current_batch = 0

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        #print(idx)
        i = idx * self.batch_size
        if i == 0:
          data_zip_list = list(zip(self.input_img_paths, self.target_img_paths))
          random.shuffle(data_zip_list)
          self.input_img_paths, self.target_img_paths = zip(*data_zip_list)
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            n = np.random.randint(0, 3)
            if n == 0:
              img = cv2.blur(img, (3, 3)) / 255.
            elif n == 1:
              img = cv2.blur(img, (5, 5)) / 255.
            else:
              img = img / 255.
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) * 1.
            y[j] = np.expand_dims(img, 2)
        return x, y

def get_model(img_size, num_classes):
  inputs = keras.Input(shape=img_size)
  
  ### [First half of the network: downsampling inputs] ###
  
  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  
  previous_block_activation = x  # Set aside residual
  
  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual
    
  ### [Second half of the network: upsampling inputs] ###
  for filters in [256, 128, 64, 32]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D(2)(x)
    
    # Project residual
    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters, 1, padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual
  
  # Add a per-pixel classification layer
  outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)
  
  # Define the model
  model = keras.Model(inputs, outputs)
  return model

tf_config = tf.ConfigProto(device_count = {'GPU': 0})
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf_config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=tf_config))

# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

# Build model
model = get_model((32, 32, 3), 1)
#model.load_weights('oxford_segmentation.h5')
model.summary()

# Split our img paths into a training and a validation set
val_samples = int(0.2*len(input_img_paths))
data_zip_list = list(zip(input_img_paths, target_img_paths))
random.shuffle(data_zip_list)
input_img_paths, target_img_paths = zip(*data_zip_list)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = Patches(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = Patches(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
opt = keras.optimizers.SGD()
model.compile(optimizer="SGD", loss="binary_crossentropy")

callbacks = [keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)]

# Train the model, doing validation at the end of each epoch.
epochs = 10
hist = model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

fig = plt.figure()
plt.plot(hist.history['loss'], label = 'Training value', color = 'darkslategray')
plt.plot(hist.history['val_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.pdf')
plt.close(fig)