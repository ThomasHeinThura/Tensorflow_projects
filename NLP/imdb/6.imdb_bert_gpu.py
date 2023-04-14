"""
The model performance : This model is so hard to train even with my Nvidia 3060
(Feature Extection performance is not looked good)
val_accuary : 50%
val_loss : 0.6962
time : 14min09sec
f1 : 0.344
epoch : 5
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras import layers
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

tf.random.set_seed(125)

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

valid_sentences = get_features_from_tfdataset(validation_data)
valid_labels = get_labels_from_tfdataset(validation_data)

train_dataset =  train_data.shuffle(15000).batch(128).prefetch(tf.data.AUTOTUNE)
valid_dataset = validation_data.batch(128).prefetch(tf.data.AUTOTUNE)
test_dataset = test_data.batch(128).prefetch(tf.data.AUTOTUNE)

# Create BERT 
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" # @param {type: "string"} ["https://tfhub.dev/google/experts/bert/wiki_books/2", "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  "https://tfhub.dev/google/experts/bert/pubmed/2", "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


preprocess_layer = hub.KerasLayer(PREPROCESS_MODEL,
                                 name="BERT_Preprocess_Layers")


bert_layer = hub.KerasLayer(BERT_MODEL,
                            trainable=False,
                            name="BERT")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

inputs = layers.Input(shape=(), dtype="string")
x = preprocess_layer(inputs)
outputs = bert_layer(x)
#x = layers.Flatten()(x)
#x = layers.Conv1D(128, 5, padding = 'same', activation = 'elu')(x)
#x = layers.Conv1D(64, 5, padding = 'same', activation = 'elu')(x)
#x = layers.GlobalMaxPool1D()(x)
#x = layers.Dense(32, activation='relu')(x)
#x = layers.Dense(32, activation='relu')(x)
#x = layers.Dense(16, activation='relu')(x)
#outputs = layers.Dense(1, activation="sigmoid")(x)
net = outputs['pooled_output']
net = tf.keras.layers.Dropout(0.1)(net)
net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
model = tf.keras.Model(inputs, net, name="BERT")

model.summary()


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)


start = datetime.now()
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=valid_dataset,
                    verbose=1)

end = datetime.now()

print(f"The time taken to train the model is :{end - start}")
results = model.evaluate(test_dataset)

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

result_preds_probs = model.predict(valid_sentences)
result_preds = tf.squeeze(tf.round(result_preds_probs))

results = calculate_accuracy_results( 
    y_true= valid_labels,
    y_pred = result_preds)

print(results)
