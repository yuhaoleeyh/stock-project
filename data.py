import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

 
start_date = datetime(2005, 9, 1)
end_date = datetime(2020, 8, 31)
 
 
dataset = []
test = []
df = web.DataReader("C31.SI", 'yahoo', start_date, end_date)
 
#invoke to_csv for df dataframe object from 
#DataReader method in the pandas_datareader library
 
#..\first_yahoo_prices_to_csv_demo.csv must not
#be open in another app, such as Excel
 
df.to_csv('capital.csv')
DAYS = 1234 #1762   #original is 2518 #google is 1510

#pulling of google data from csv file
capital = pd.read_csv('D:/didi/ml/stock-project/capital.csv')

 
history_points = 50

 
stocks = capital
 
 
stocks.drop(columns = ['Date'], inplace = True, axis = 1)
 
test_split = 0.6 # the percent of data to be used for testing
n = int(stocks.shape[0] * test_split)

print(n)


decomposition = seasonal_decompose(stocks['Adj Close'], period = 253 )

#253 trading days in a year

season = decomposition.seasonal
decomposition.plot()
plt.show()
for i in range(len(season)):
    stocks['Adj Close'][i] -= season[i] 

train_data = stocks[:n]



 
#normalise data so that various columns will have equal weight
normaliser = preprocessing.MinMaxScaler()
normalised_data = normaliser.fit_transform(train_data)


# print(normalised_data[:,5])

#preparing of input and output
ohlcv_histories_normalised = np.array([normalised_data[:,5][i  : i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.array([normalised_data[:,5][i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
 
 
next_day_open_values = np.array([train_data['Adj Close'][i + history_points].copy() for i in range(len(train_data) - history_points)])
 
 
next_day_open_values = np.expand_dims(next_day_open_values, -1)
 

#normalise y so we can do an inverse later on once we test the result
y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(next_day_open_values)
 
 
assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
 
#same thing is done for test data. note test and train data must always be separate from one another, ie. the model should never have the test set during the training phase 
test_data = stocks[n:]
  
 
test_normalised_data = normaliser.transform(test_data)
 
ohlcv_histories_normalised_test = np.array([test_normalised_data[:,5][i  : i + history_points].copy() for i in range(len(test_normalised_data) - history_points)])
next_day_open_values_normalised_test = np.array([test_normalised_data[:,5][i + history_points].copy() for i in range(len(test_normalised_data) - history_points)])
next_day_open_values_normalised_test = np.expand_dims(next_day_open_values_normalised_test, -1)
 
 
next_day_open_values_test = np.array([test_data['Adj Close'][DAYS + i + history_points].copy() for i in range(len(test_data) - history_points)])
 
next_day_open_values_test = np.expand_dims(next_day_open_values_test, -1)
 
assert ohlcv_histories_normalised_test.shape[0] == next_day_open_values_normalised_test.shape[0]
 
ohlcv_train = ohlcv_histories_normalised
y_train = next_day_open_values_normalised
 
ohlcv_test =  ohlcv_histories_normalised_test
y_test = next_day_open_values_normalised_test
 
unscaled_y_test = next_day_open_values_test
 

#prepare second feature, exponential moving average
 
exponential_moving_average = [] #factors in more weight to recent data rather than a simple moving average which weighs everything equally (not good for stock data )
 
multiplier = 2.0 / 51

sma = np.mean(ohlcv_train[0])
 
exponential_moving_average.append(np.array([sma]))
 
for i in range(1, len(ohlcv_train)):
    sma = (train_data['Adj Close'][i] - sma) * multiplier + sma
    exponential_moving_average.append(np.array([sma]))
 
scaler = preprocessing.MinMaxScaler()
exponential_moving_average_normalised = scaler.fit_transform(exponential_moving_average)
 
assert exponential_moving_average_normalised.shape[0] == y_train.shape[0]
 
 #second feature for test data
exponential_moving_test = []
 
 
sma = np.mean(ohlcv_test[0])
 
exponential_moving_test.append(np.array([sma]))
 
for i in range(1, len(ohlcv_test)):
    sma = (test_data['Adj Close'][i + DAYS] - sma) * multiplier + sma
    exponential_moving_test.append(np.array([sma]))
 
exponential_moving_test_normalised = scaler.transform(exponential_moving_test)
 
assert exponential_moving_test_normalised.shape[0] == y_test.shape[0]

 
# machine learning libraries to import 
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.callbacks import History 
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
 
# print(ohlcv_histories_normalised.shape[1]) 
# print(exponential_moving_average_normalised.shape[1])
#prepare the 2 features to be inputted into model 
lstm_input = Input(shape=(history_points, 1), name='lstm_input')
exponential_input = Input(shape=(exponential_moving_average_normalised.shape[1],), name='ema_input')
 
#first branch
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)
 
# the second branch opreates on the second input
y = Dense(64, name='avg_dense_0')(exponential_input)
y = Activation("relu", name='avg_relu_0')(y)
y = Dropout(0.2, name='avg_dropout_0')(y)
average_branch = Model(inputs=exponential_input, outputs=y)
 
 
# combine the output of the two branches
combined = concatenate([lstm_branch.output, average_branch.output], name='concatenate')
 
z = Dense(128, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)
 
# our model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[lstm_branch.input, average_branch.input], outputs=z)
 
 
adam = optimizers.Adam(lr=0.0005)
 
model.compile(optimizer=adam, loss='mse')
#fitting of model. note the validation split of 0.1 
model.fit(x=[ohlcv_train, exponential_moving_average_normalised], y=y_train, batch_size=32, epochs=150, shuffle=True, validation_split = 0.1) #callbacks = [history]
 

#predicting of model against test 
y_test_predicted = model.predict([ohlcv_test, exponential_moving_test_normalised])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)
 
plt.gcf().set_size_inches(22, 15, forward=True)
 
start = 0
end = -1
 
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# decomposition = seasonal_decompose(stocks['Adj Close'], freq = 365) 
# season = plt.plot(decomposition, label = "Seasonality")

#plotting of model  
plt.legend(['Real', 'Predicted'])
 
plt.show()


