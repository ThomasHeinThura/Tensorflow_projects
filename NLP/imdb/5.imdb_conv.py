"""
The model performance :
val_accuary : 86.11%
val_loss : 0.3410
time : 1min40sec
f1 : 0.8611
epoch : 10
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras import layers 
from tensorflow.keras.layers import TextVectorization 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


max_vocab = 5000  # Maximum vocab size.
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
vectorize_layer.adapt(features)

embedding_layers = layers.Embedding(input_dim=max_vocab,
                                     output_dim=5,
                                     embeddings_initializer="uniform",
                                     input_length = max_seq_len,
                                     name="embedding_layers")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

inputs = layers.Input(shape=(1,), dtype="string")
x = vectorize_layer(inputs)
x = embedding_layers(x)
x = layers.Conv1D(128, 5, padding = 'same', activation = 'elu')(x)
#x = layers.Conv1D(64, 5, padding = 'same', activation = 'elu')(x)
x = layers.GlobalMaxPool1D()(x)
#x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs, name="Conv")

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00065),
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

result_preds_probs = model.predict(valid_features)
result_preds = tf.squeeze(tf.round(result_preds_probs))

results = calculate_accuracy_results( 
    y_true= valid_labels,
    y_pred = result_preds)

print(results)
