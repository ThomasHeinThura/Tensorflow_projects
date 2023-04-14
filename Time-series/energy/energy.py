"""
The model performance : 
MAE : 0.0009
MSE : 0.0111
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


# Import Data Main data from keggle and tensorflow dataset
# import data by pointing exact location or os.path.dirname(os.path.realpath('example-file/sign-mnist/sign_mnist_train.csv'))
dir_path = '/home/hanlinn/00.projects/tensorflow-prepare/example-file/energy/PJME_hourly.csv'


# import dataset
df = pd.read_csv(dir_path)
#Remove datetime
date_time = pd.to_datetime(df.pop('Datetime'), format='%Y-%m-%d %H:%M:%S')

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

#multi_window.plot()
print(multi_window)

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['mae'])

val_performance = {}
performance = {}

val_performance['Last'] = last_baseline.evaluate(multi_window.val)
performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
#multi_window.plot(last_baseline)


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(lstm_model, multi_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate( multi_window.val)
performance['LSTM'] = lstm_model.evaluate( multi_window.test, verbose=0)

print(performance)