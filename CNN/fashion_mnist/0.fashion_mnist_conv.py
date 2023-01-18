"""
The model performance
val_accurary : 90.21%
val_loss : 0.27
time : 2min29sec
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers
from datetime import datetime

print(tf.__version__)
tf.get_logger().setLevel('ERROR')
# Get data
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize images (get values between 0 & 1)
train_images, test_images = train_images / 255.0, test_images / 255.0 

# Check shape of input data
# print(train_images.shape)
# print(train_labels.shape)

# Build model
model = tf.keras.Sequential([
    # Reshape inputs to be compatible with Conv2D layer
    layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(), # flatten outputs of final Conv layer to be suited for final Dense layer
    layers.Dense(10, activation="softmax")
])

# Compile model 
model.compile(loss="sparse_categorical_crossentropy", # if labels aren't one-hot use sparse (if labels are one-hot, drop sparse)
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

start = datetime.now()
# Fit model
print("Training model...")
model.fit(x=train_images,
          y=train_labels,
          epochs=10,
          validation_data=(test_images, test_labels))
end = datetime.now()
print(f'The time taken to train the model is {end-start}')
# Evaluate model 
print("Evaluating model...")
model.evaluate(test_images, test_labels)

# Save model to current working directory
model.save("test_image_model.h5")


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

model_preds_probs = model.predict(test_images)
model_preds = tf.argmax(model_preds_probs, axis=1)

model_result = calculate_accuracy_results(y_pred=model_preds,
                                           y_true=test_labels)

print(model_result)
