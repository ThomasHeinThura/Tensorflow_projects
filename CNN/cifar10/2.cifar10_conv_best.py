"""
The model performance : 
val_accuary : 86%
val_loss : 0.4491
time : 53min
epoch : 25 (get val_accuary 77.78% on 10 epoch and loss is 0.6628 time is 22min51sec)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.set_seed = 42
epoch = 25
input_shape = (32, 32, 3)

# import data
(train_features,train_labels), (test_features, test_labels) = keras.datasets.cifar10.load_data()

train_features = tf.cast(train_features, dtype=tf.float32) / 255
test_features = tf.cast(test_features, dtype=tf.float32) / 255
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

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
train_dataset =  train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features,test_labels))
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

#Build the model
inputs = layers.Input(shape=input_shape, name="input_layer")
x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(inputs)
x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, kernel_size=3, padding="same" ,activation="elu")(x)
x = layers.Conv2D(128, kernel_size=3, padding="same",activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="elu", name="Dense_1")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="elu", name="Dense_2")(x)
outputs = layers.Dense(10, activation="softmax",name="output_layer")(x)      
model = tf.keras.Model(inputs, outputs) 

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),
              metrics=["accuracy"])

model.summary()

start = datetime.now()
history_model = model.fit(train_dataset,
                          steps_per_epoch=len(train_dataset),
                          validation_data=valid_dataset,
                          validation_steps=int(0.25*len(valid_dataset)),
                          callbacks=[early_stopping, reduce_lr],
                          epochs=epoch) 
end = datetime.now()

print(f"The time taken to train the model is {end - start}")
# Evaluate model
model.evaluate(valid_dataset)

def calculate_accuracy_results(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    """
     Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division= 1)
    model_results = {"accuracy": model_accuracy,
                      "precision": model_precision,
                      "recall": model_recall,
                      "f1": model_f1}
    return model_results

model_preds_probs = model.predict(test_features)
model_preds = tf.argmax(model_preds_probs, axis=1)
test_labels_encode = tf.argmax(test_labels,axis=1)
model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=test_labels_encode)
print(model_result)
