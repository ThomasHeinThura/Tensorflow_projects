"""
The model performace : I cheat alot in this dataset
val_accuracy : 81% 
val_loss : 0.9033
Time : 3min20sec
epoch : 5
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import pandas as pd

from datetime import datetime


tf.get_logger().setLevel('ERROR')
#tf.autograph.set_verbosity(1)
tf.set_seed = 42
epoch = 10
max_vocab_length = 10000 # max number of words to have in our vocabulary
#max_length = 100

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.reuters.load_data(num_words=max_vocab_length)

# Check the data shape
print(
    f"Train_features : {train_features.shape} {train_features.dtype} \n" 
    f"Train_labels : {train_labels.shape} {train_labels.dtype} \n" 
    f"Test features : {test_features.shape} {test_features.dtype} \n" 
    f"Test_labels : {test_labels.shape} {test_labels.dtype} "
    )

# VECTORIZE function

def vectorize_sequences(sequences, dimension=max_vocab_length):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return tf.cast(results, dtype= tf.float32)

# Vectorize and Normalize train and test to tensors with 10k columns

train_features_tf = vectorize_sequences(train_features)
test_features_tf = vectorize_sequences(test_features)

print("train_features ", train_features_tf.shape)
print("test_Features ", test_features_tf.shape)

# ONE HOT ENCODER of the labels

one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_features_tf, one_hot_train_labels))
train_dataset =  train_dataset.shuffle(8982).batch(32).cache().prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_features_tf,one_hot_test_labels))
valid_dataset = valid_dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=2) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=1,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-6)

embedding_layers = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=2,
                                     name="embedding_layers")

inputs = layers.Input(shape=(max_vocab_length,))
x = embedding_layers(inputs)
x = layers.Conv1D(16,5, padding='same', activation='elu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.6)(x)
x = tf.keras.layers.Dense(60, activation='elu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(46, activation="softmax")(x)
model= tf.keras.Model(inputs, outputs, name="Dense_model")

model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate = 0.00075
                    ),
                metrics=["accuracy"])

model.summary()

start = datetime.now()
model_history = model.fit(train_dataset,
                           epochs=5,
                           validation_data=valid_dataset,
                           callbacks=[early_stopping, reduce_lr])

end = datetime.now()
print(f"The time taken to fit the modle is {end - start}")
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

model_preds_probs = model.predict(test_features_tf)
model_preds = tf.argmax(model_preds_probs, axis=1)
test_labels_encode = tf.argmax(one_hot_test_labels ,axis=1)

model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=test_labels_encode)
print(model_result)