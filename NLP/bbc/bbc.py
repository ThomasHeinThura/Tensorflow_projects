"""
val_accurary : 90%
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

data_folder = '/home/hanlinn/00.projects/tensorflow-prepare/example-file/bbc/'

train_csv_path = data_folder + 'BBC News Train.csv'
test_csv_path = data_folder + 'BBC News Test.csv'
sample_solutions_path = data_folder + 'BBC News Sample Solution.csv'

train_df = pd.read_csv(train_csv_path)
train_df.head()
train_df.dropna()

from sklearn.model_selection import train_test_split
train_sentence,val_sentence,train_label,val_labels = train_test_split(train_df["Text"].to_numpy(),
                                                                      train_df["Category"].to_numpy(),
                                                                      test_size=0.1, # dedicate 10% of samples to validation set
                                                                       random_state=42)

print(train_sentence.shape, val_sentence.shape, train_label.shape, val_labels.shape)

train_label_cate = pd.get_dummies(train_label)
trian_label_one_hot = train_label_cate.to_numpy()

val_label_cate = pd.get_dummies(val_labels)
val_label_one_hot = val_label_cate.to_numpy()

train_dataset = tf.data.Dataset.from_tensor_slices((train_sentence, trian_label_one_hot))
train_dataset =  train_dataset.shuffle(1341).batch(32).cache().prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentence,val_label_one_hot))
valid_dataset = valid_dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)
print(f"Train : {train_dataset} \n"
      f"Test : {valid_dataset}")
                                                        

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
vectorize_layer.adapt(train_sentence)

embedding_layers = tf.keras.layers.Embedding(input_dim=max_vocab,
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

from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = vectorize_layer(inputs)
x = embedding_layers(x)
x = layers.Conv1D(64, 5, padding = 'same', activation = 'elu')(x)
x = layers.Conv1D(128, 5, padding = 'same', activation = 'elu')(x)
x = layers.GlobalMaxPool1D()(x)
# x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="Conv")

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00065),
              loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

from datetime import datetime

start = datetime.now()
history = model.fit(train_dataset,
                    epochs= 20,
                    validation_data=valid_dataset,
                    verbose=1)

end = datetime.now()

print(f"The time taken to train the model is :{end - start}")
results = model.evaluate(valid_dataset)



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

result_preds_probs = model.predict(val_sentence)
result_preds = tf.argmax(result_preds_probs, axis=1)
val_labels_encode = tf.argmax(val_label_one_hot ,axis=1)
results = calculate_accuracy_results( 
    y_true= val_labels_encode,
    y_pred = result_preds)

print(results)

