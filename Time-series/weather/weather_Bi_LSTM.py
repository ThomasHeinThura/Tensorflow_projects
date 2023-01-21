"""
The model performance : 
MSE : 0.001413
MAE : 0.02186
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from windows import WindowGenerator, MultiStepLastBaseline, compile_and_fit
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# import dataset
df = pd.read_csv(csv_path)
#Remove datetime
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
# fix error
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
# fix error
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

#Change wide direction degree to sin and cos
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

# change time to sin and cos
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

#Check and normalize
print(f"The original dataset shape is :{df.shape}")
from sklearn.preprocessing import minmax_scale
scaled_df = minmax_scale(df)
print(f"The scaled dataset shape is : {scaled_df.shape}")

# check column name
column_names = list(df.columns.values)
scaled_df = pd.DataFrame(scaled_df.tolist(),columns=column_names)

#Train_test_split
n = len(scaled_df)
train_df = scaled_df[0:int(n*0.7)]
val_df = scaled_df[int(n*0.7):int(n*0.9)]
test_df = scaled_df[int(n*0.9):]

num_features = scaled_df.shape[1]
print(f"The train shape is : {train_df.shape} \n"
      f"The valid shape is : {val_df.shape} \n"
      f"The test shape is : {test_df.shape} \n"
      f"The number of feature: {num_features}")


OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df= train_df,
                               val_df= val_df,
                               test_df= test_df)
val_performance = {}
performance = {}

#multi_window.plot()
print(multi_window)

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small.
        # Therefore, initialize the output layer with zeros.
        kernel_initializer=tf.initializers.zeros())
]))

history = compile_and_fit(residual_lstm, multi_window)

IPython.display.clear_output()
val_performance['LSTM'] = residual_lstm.evaluate( multi_window.val)
performance['LSTM'] = residual_lstm.evaluate( multi_window.test, verbose=0)

print(performance)