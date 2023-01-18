"""
The model performace
val_accuary = 99.14%
val_loss = 0.0245
time = 1:10
"""

# Import all required libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf

# Split MNIST Train and Test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Divide train set by 255
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train, y_test = keras.utils.to_categorical(y_train, num_classes=10), keras.utils.to_categorical(y_test, num_classes=10)
# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
# Model summary and Evaluation
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
start = datetime.now()
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
stop = datetime.now()
print("Time taken to execute:" + str(stop - start))

model.evaluate(x_test, y_test)

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

model_preds_probs = model.predict(x_test)
model_preds = tf.argmax(model_preds_probs, axis=1)
y_test_encode = tf.argmax(y_test,axis=1)
model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=y_test_encode)
print(model_result)