"""
val_accuary : 98%
val_loss : 0.0488
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Import Data Main data from keggle and import data by pointing exact location or os.path.dirname(os.path.realpath('example-file/sign-mnist/sign_mnist_train.csv'))
data_path = '/home/hanlinn/00.projects/tensorflow-prepare/example-file/sign-mnist/sign_mnist_train.csv'
train_df = pd.read_csv(data_path)

test_data_path = '/home/hanlinn/00.projects/tensorflow-prepare/example-file/sign-mnist/sign_mnist_test.csv'
test_df = pd.read_csv(test_data_path)

train_label = train_df['label']
test_label = test_df['label']
del train_df['label']
del test_df['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
train_label = label_binarizer.fit_transform(y=train_label)
test_label = label_binarizer.fit_transform(y=test_label)

train_feature = train_df.values
test_feature = test_df.values
reshape_train = train_feature / 255
reshape_test = test_feature / 255

reshape_train = reshape_train.reshape(-1,28,28,1)
reshape_test = reshape_test.reshape(-1,28,28,1)

# Preprocess the data
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((reshape_train, train_label))
train_dataset =  train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((reshape_test,test_label))
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


from tensorflow.keras import layers
input_shape =(28,28,1)
#Build the model
inputs = layers.Input(shape=input_shape, name="input_layer")
x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(inputs)
# x = layers.Conv2D(32, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
# x = layers.Conv2D(64, kernel_size=3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, kernel_size=3, padding="same" ,activation="elu")(x)
# x = layers.Conv2D(128, kernel_size=3, padding="same",activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="elu", name="Dense_1")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="elu", name="Dense_2")(x)
outputs = layers.Dense(24, activation="softmax",name="output_layer")(x)      
model = tf.keras.Model(inputs, outputs) 

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

from datetime import datetime
epoch = 10
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

model_preds_probs = model.predict(reshape_test)
model_preds = tf.argmax(model_preds_probs, axis=1)
test_labels_encode = tf.argmax(test_label,axis=1)
model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=test_labels_encode)
print(model_result)

