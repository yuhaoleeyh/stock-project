import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np


start_date = datetime(2010, 9, 1)
end_date = datetime(2020, 8, 31)



apple = pd.read_csv('./apple_stocks_data.csv')
# google = pd.read_csv('./google_stocks_data.csv')

history_points = 50

stocks = apple


stocks.drop(columns = ['Date'], inplace = True, axis = 1)

test_split = 0.7 # the percent of data to be used for testing
n = int(stocks.shape[0] * test_split)

train_data = stocks[:n]

normaliser = preprocessing.MinMaxScaler()
normalised_data = normaliser.fit_transform(train_data)

ohlcv_histories_normalised = np.array([normalised_data[i  : i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.array([normalised_data[:,2][i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

next_day_open_values = np.array([stocks['Open'][i + history_points].copy() for i in range(len(stocks) - history_points)])
next_day_open_values = np.expand_dims(next_day_open_values, -1)

# print(next_day_open_values)

y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(next_day_open_values)


assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]

test_data = stocks[n:]


maximum_train = 100
minimum_train = 0 #widen the space of the train data to accomodate the testing data, scaling factor. 
print(maximum_train)
print(minimum_train)

test_original = test_data

test_data = np.array(test_data['Open'])
test_data -= minimum_train
test_data /= (maximum_train - minimum_train) 
print(test_original['Open'])

ohlcv_test_normalised = np.array([test_data[i  : i + history_points].copy() for i in range(len(test_data) - history_points)])
next_day_open_values_normalised_test = np.array([test_data[i + history_points].copy() for i in range(len(test_data) - history_points)])
next_day_open_values_normalised_test = np.expand_dims(next_day_open_values_normalised_test, -1)



next_day_open_values_test = np.array([test_original['Open'][i + history_points].copy() for i in range(len(test_original) - history_points)])
next_day_open_values_test = np.expand_dims(next_day_open_values_test, -1)



# splitting the dataset up into train and test sets

ohlcv_train = ohlcv_histories_normalised
y_train = next_day_open_values


ohlcv_test = ohlcv_test_normalised
y_test = next_day_open_values_test

unscaled_y_test = next_day_open_values_test

# print(ohlcv_test.shape) #(50, 6)
# print(y_test.shape) #(1,)






# import keras
# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
# from keras import optimizers
# import numpy as np
# np.random.seed(4)
# tf.random.set_seed(4)

# lstm_input = Input(shape=(history_points, 6), name='lstm_input')
# x = LSTM(50, name='lstm_0')(lstm_input)
# x = Dropout(0.2, name='lstm_dropout_0')(x)
# x = Dense(64, name='dense_0')(x)
# x = Activation('sigmoid', name='sigmoid_0')(x)
# x = Dense(1, name='dense_1')(x)
# output = Activation('linear', name='linear_output')(x)
# model = Model(inputs=lstm_input, outputs=output)

# adam = optimizers.Adam(lr=0.0005)

# model.compile(optimizer=adam, loss='mae')

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model.png')


# model.fit(x=ohlcv_train, y=y_train, batch_size=100, epochs=50, shuffle=True, validation_split=0.1)
# evaluation = model.evaluate(ohlcv_test, y_test)
# print(evaluation)


# y_test_predicted = model.predict(ohlcv_test)
# print(y_test_predicted)
# y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
# y_predicted = model.predict(ohlcv_histories_normalised)
# y_predicted = y_normaliser.inverse_transform(y_predicted)

# assert unscaled_y_test.shape == y_test_predicted.shape
# real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
# scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
# print(scaled_mse)

# plt.gcf().set_size_inches(22, 15, forward=True)

# start = 0
# end = -1

# real = plt.plot(unscaled_y_test[start:end], label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')

# # real = plt.plot(unscaled_y[start:end], label='real')
# # pred = plt.plot(y_predicted[start:end], label='predicted')

# plt.legend(['Real', 'Predicted'])

# plt.show()
