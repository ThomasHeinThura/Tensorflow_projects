"""
The model performace : 
val_accuracy : 59.97%
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
#tf.autograph.set_verbosity(1)
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow import keras
from datetime import datetime
import pandas as pd
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


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


#### One hot
one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

#from sklearn.preprocessing import OneHotEncoder
#one_hot_encoder = OneHotEncoder(sparse=False)
#train_labels_one_hot = one_hot_encoder.fit_transform(train_labels.numpy().reshape(-1, 1))
#test_labels_one_hot = one_hot_encoder.transform(test_labels.numpy().reshape(-1, 1))

# Extract labels ("target" columns) and encode them into integers 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)


# Check what training labels look like
#print(train_labels_encoded)

train_sentences = train_decode.numpy().tolist()
test_sentences = test_decode.numpy().tolist()
print(len(train_sentences), len(test_sentences))


# Create a pipeline
model_0 = Pipeline([
  ("tf-idf", TfidfVectorizer()),
  ("clf", MultinomialNB())
])

# Fit the pipeline to the training data
model_0.fit(X=train_sentences, 
            y=train_labels_encoded);

# Evaluate baseline on validation dataset
model_0.score(X=test_sentences,
              y=test_labels_encoded)

baseline_preds = model_0.predict(test_sentences)
#print(baseline_preds)

def calculate_accuracy_results(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division= 1)
    model_results = {"accuracy": model_accuracy,
                      "precision": model_precision,
                      "recall": model_recall,
                      "f1": model_f1}
    return model_results

baseline_results = calculate_accuracy_results(test_labels_encoded,baseline_preds)
print(baseline_results)