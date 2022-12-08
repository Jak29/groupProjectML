# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:29:53 2022

@author: culli
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

# Load Training Data
company = "BTC-USD"
start = dt.datetime(2015,1,1)
end = dt.datetime(2022,1,1)
data = web.DataReader(company, "yahoo", start, end)


# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=64)

# Load Data
test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, "yahoo", test_start, test_end)
actual_prices = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis = 0)



model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values # Important
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Loop for making predictions
for x in range(prediction_days):
    
    
    # Make Predictions on Test Data
    x_test = []
    
    for x in range(prediction_days, len(model_inputs)+1):
        x_test.append(model_inputs[x-prediction_days:x, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    
    # Predict Next Day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    
    # Formatting the prediction and printing it
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")
    
    # Adding the prediction back into the model for the next prediction
    model_prediction_input = prediction
    model_prediction_input = model_prediction_input.reshape(-1, 1)
    model_prediction_input = scaler.transform(model_prediction_input)
    model_inputs = np.concatenate((model_inputs, model_prediction_input))


plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Days")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()


tf.keras.models.save_model(model,"model.pbtxt")
converter = lite.TFLiteConverter.from_keras_model(model = model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS]
model_tflite = converter.convert()
open("BTCPrediction.tflite", "wb").write(model_tflite)

