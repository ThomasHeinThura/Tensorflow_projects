"""
The model performance :
val_accuary : 85.94%
val_loss : 0.3274
time : 3min38secs
f1 : 0.8594
epoch : 10
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras import layers 
from tensorflow.keras.layers import TextVectorization 


tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
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


labels = get_labels_from_tfdataset(train_data)
features = get_features_from_tfdataset(train_data)
valid_features = get_features_from_tfdataset(validation_data)
valid_labels = get_labels_from_tfdataset(validation_data)
print(valid_features.shape, valid_labels.shape)
print(features.shape, labels.shape)

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[], # shape of inputs coming to our model 
                                        dtype=tf.string, # data type of inputs coming to the USE layer
                                        trainable=False, # keep the pretrained weights (we'll create a feature extractor)
                                        name="USE")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

model = tf.keras.Sequential([
  sentence_encoder_layer, # take in sentences and then encode them into an embedding
  layers.Dense(64, activation="relu"),
  layers.Dense(1, activation="sigmoid")
], name="model_USE")

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()

start = datetime.now()
history = model.fit(features,
                    labels,
                    batch_size=128,
                    epochs=epoch,
                    validation_data=[valid_features, valid_labels],
                    verbose=1)

end = datetime.now()

print(f"The time taken to train the model is :{end - start}")

results = model.evaluate(valid_features, valid_labels)

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
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                      "precision": model_precision,
                      "recall": model_recall,
                      "f1": model_f1}
    return model_results

result_preds_probs = model.predict(valid_features)
result_preds = tf.squeeze(tf.round(result_preds_probs))

results = calculate_accuracy_results( 
    y_true= valid_labels,
    y_pred = result_preds)

print(results)
