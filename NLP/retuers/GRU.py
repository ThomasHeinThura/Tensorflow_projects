"""
The model performace : 
val_accuracy : 66.60%
val_loss : 1.4097
time : 7min1sec
f1 : 0.6376
epochs : 10
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
#tf.autograph.set_verbosity(1)
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import layers
from tensorflow import keras
from datetime import datetime
import pandas as pd
import tensorflow_hub as hub

# import data
(train_features,train_labels), (test_features, test_labels) = tf.keras.datasets.reuters.load_data()

#Reverse word index
word_index = tf.keras.datasets.reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
len(reverse_word_index)

def decode_tokenized_sentence(sentences_array):
    decode_sent = []
    for values in sentences_array:
        decoded_newswire = [' '.join([reverse_word_index.get( i - 3, '?') for i in values ]) ]
        decode_sent.append(decoded_newswire)
    return tf.constant(tf.squeeze(decode_sent))

train_decode = decode_tokenized_sentence(train_features)
test_decode = decode_tokenized_sentence(test_features)
print(f" Train : {train_decode.shape} \n" 
      f"Test : {test_decode.shape}")

one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_decode, one_hot_train_labels))
train_dataset =  train_dataset.shuffle(8982).batch(128).cache().prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((test_decode,one_hot_test_labels))
valid_dataset = valid_dataset.batch(128).cache().prefetch(tf.data.AUTOTUNE)
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

max_vocab = 10_000  # Maximum vocab size.
max_seq_len = 600  # Sequence length to pad the outputs to.

# Create the layer.
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_vocab,
    output_mode='int',
    output_sequence_length=max_seq_len)

# Now that the vocab layer has been created, call `adapt` on the
# text-only dataset to create the vocabulary. You don't have to batch,
# but for large datasets this means we're not keeping spare copies of
# the dataset.
vectorize_layer.adapt(train_decode)

test_vectorize = vectorize_layer(train_decode)

print(f"The testing for vectorize {test_vectorize.shape}")
embedding_layers = layers.Embedding(input_dim=max_vocab,
                                     output_dim=32,
                                     embeddings_initializer="uniform",
                                     input_length = max_seq_len,
                                     name="embedding_layers")

model_hub = tf.keras.Sequential([
  layers.Input(shape=(1,), dtype=tf.string),
  vectorize_layer,
  embedding_layers,
#   tf.keras.layers.Bidirectional(layers.GRU(64, return_sequences=True)),
  tf.keras.layers.Bidirectional(layers.GRU(128)),
#   layers.Dense(256, activation="relu"),
  layers.Dense(128, activation="relu"),
  layers.Dense(46, activation="softmax")
], name="GRU")

# Compile
model_hub.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_hub.summary()


start = datetime.now()
model_USE_history = model_hub.fit(train_dataset,
                                  epochs=10,
                                  validation_data=valid_dataset,
                                  callbacks=[early_stopping, reduce_lr])

end = datetime.now()
print(f"The time taken to fit the modle is {end - start}")
model_hub.evaluate(valid_dataset)


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

model_preds_probs = model_hub.predict(test_decode)
model_preds = tf.argmax(model_preds_probs, axis=1)

model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=test_labels)
print(model_result)