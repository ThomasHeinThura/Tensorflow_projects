import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.set_seed = 42
epoch = 25
input_shape = (32, 32, 3)


# import data
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.cifar100.load_data()

train_features = tf.cast(train_features, dtype=tf.float32) / 255
test_features = tf.cast(test_features, dtype=tf.float32) / 255
train_labels = keras.utils.to_categorical(train_labels, num_classes=100)
test_labels = keras.utils.to_categorical(test_labels, num_classes=100)


# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    ) 

# Preprocess the data
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset =  train_dataset.shuffle(50000).batch(128).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
valid_dataset = valid_dataset.batch(128).prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=15) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=5,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# Setup input shape and base model, freezing the base model layers

#Build the model
inputs = layers.Input(shape=input_shape, name="input_layer")
x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(256, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(256, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(512, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(512, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax",name="output_layer")(x)      
model = tf.keras.Model(inputs, outputs) 

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

start = datetime.now()
history_model = model.fit(train_dataset,
                          steps_per_epoch=len(train_dataset),
                          validation_data=valid_dataset,
                          validation_steps=int(0.1*len(valid_dataset)),
                          callbacks=[early_stopping, reduce_lr],
                          epochs=epoch) 
end = datetime.now()
print(f"The time taken to train the model is {end - start}")
# Evaluate model
model.evaluate(valid_dataset)
