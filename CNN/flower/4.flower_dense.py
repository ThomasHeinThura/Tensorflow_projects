"""
The model performance : 
val_accuary : 42%
val_loss : 1.3371
time : 3min1sec
epoch : 10
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import PIL.Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
import tensorflow_datasets as tfds
import pathlib
from datetime import datetime

batch_size = 32
img_height = 128
img_width = 128
AUTOTUNE = tf.data.AUTOTUNE
input_shape = (img_height,img_width, 3)
num_classes = 5
epoch = 10

# Import Data Main data from keggle or from tensorflow dataset and import data.
flower_dir =  os.path.dirname(os.path.realpath('example-file/flower/flowers/*'))

train_ds = tf.keras.utils.image_dataset_from_directory(
  flower_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  flower_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print(train_ds)
print(val_ds)
class_names = train_ds.class_names
print(class_names)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

#Build model
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(input_shape),name='input_layers'),
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(258, activation='elu'),
  tf.keras.layers.Dense(128, activation='elu'),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(64, activation='elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

model.summary()

start = datetime.now()
history_model = model.fit(train_ds,
                          steps_per_epoch=len(train_ds),
                          validation_data=val_ds,
                          validation_steps=int(0.25*len(val_ds)),
                          callbacks=[early_stopping, reduce_lr],
                          epochs=epoch) 
end = datetime.now()


print(f"The time taken to train the model is {end - start}")
# Evaluate model
model.evaluate(val_ds)

