"""
The model performance : 
val_accuary : 87.22% 
val_loss : 0.3542
time : 2min54sec
f1: 0.8722
epoch : 10
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from datetime import datetime

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

tf.set_seed = 42
epoch = 10


# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Data preparation 

def get_features_from_tfdataset(tfdataset, batched=False):

    features = list(map(lambda x: x[0], tfdataset)) # Get labels 

    if not batched:
        return tf.stack(features, axis=0) # concat the list of batched labels

    return features

def get_labels_from_tfdataset(tfdataset, batched=False):

    labels = list(map(lambda x: x[1], tfdataset)) # Get labels 

    if not batched:
        return tf.stack(labels, axis=0) # concat the list of batched labels

    return labels

valid_sentence = get_features_from_tfdataset(validation_data)
valid_labels = get_labels_from_tfdataset(validation_data)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),)
model.add(tf.keras.layers.Conv1D(16,3,activation='elu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


start = datetime.now()
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=epoch,
                    validation_data=validation_data.batch(512),
                    verbose=1)

end = datetime.now()

print(f"The time taken to train the model is :{end - start}")
results = model.evaluate(test_data.batch(512))
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

model_preds_probs = model.predict(valid_sentence)
model_preds = tf.squeeze(tf.round(model_preds_probs))

model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=valid_labels)
print(model_result)
