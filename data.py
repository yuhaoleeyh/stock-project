import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
 
# import keras
# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
# from keras import optimizers
 
 
start_date = datetime(2005, 9, 1)
end_date = datetime(2020, 8, 31)
 
# stocks = ['INPX', 'AAPL', 'MSFT', '^GSPC']
 
dataset = []
test = []
 
panel_data = data.DataReader('MSFT', start = start_date, end = end_date, data_source = 'yahoo')
panel_data.to_csv('./microsoft_stocks_data.csv')
# google = pd.read_csv('./google_stocks_data.csv')
# apple = pd.read_csv('./apple_stocks_data.csv')
microsoft = pd.read_csv('./microsoft_stocks_data.csv')
# gspc = pd.read_csv('./gspc_stocks_data.csv')
#
 
history_points = 50
 
# print(panel_data)
# dataset.append(panel_data.assign(i = i)[['Adj Close']])
    # print(panel_data.shape)
    # print(panel_data)
    # print(stocks[i])
 
# panel_data.to_csv('./stocks_data.csv')
 
stocks = microsoft
 
# for i in range(len(stocks)):
#     print(stocks[i].head())
 
 
stocks.drop(columns = ['Date'], inplace = True, axis = 1)
# print(stocks.head())
 
# for i in range(len(stocks)):
#     print(stocks[i].shape)
 
# for i in range(len(stocks)):
#     print(stocks[i].describe())
 
# for i in range(len(stocks)):
#     lists = [0]
#     for j in range(1, len(stocks[i]['Adj Close'])):
#         lists.append((stocks[i]['Adj Close'][j] - stocks[i]['Adj Close'][j - 1]) * 100.0 / stocks[i]['Adj Close'][j - 1]) 
#     stocks[i]['Percentage change'] = lists
#     print(stocks[i][stocks[i]['Percentage change'] > 15])
 
 
# print(stocks.shape[0])
 
test_split = 0.7 # the percent of data to be used for testing
n = int(stocks.shape[0] * test_split)
 
train_data = stocks[:n]

print(train_data)
 
normaliser = preprocessing.MinMaxScaler()
normalised_data = normaliser.fit_transform(train_data)
 
    # for j in range(5):
    #     print(normalised_data[j])
# print(normalised_data[:,0])
# print(stocks['High'][0:4])
 
ohlcv_histories_normalised = np.array([normalised_data[i  : i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.array([normalised_data[:,2][i + history_points].copy() for i in range(len(normalised_data) - history_points)])
next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
 
 
next_day_open_values = np.array([train_data['Open'][i + history_points].copy() for i in range(len(train_data) - history_points)])
 
 
next_day_open_values = np.expand_dims(next_day_open_values, -1)
 
 
 
# print(next_day_open_values)
# print(normalised_data[:,2])
 
y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(next_day_open_values)
 
 
assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
 
test_data = stocks[n:]
 
# print(test_data)
 
 
test_normalised_data = normaliser.transform(test_data)
 
ohlcv_histories_normalised_test = np.array([test_normalised_data[i  : i + history_points].copy() for i in range(len(test_normalised_data) - history_points)])
next_day_open_values_normalised_test = np.array([test_normalised_data[:,2][i + history_points].copy() for i in range(len(test_normalised_data) - history_points)])
next_day_open_values_normalised_test = np.expand_dims(next_day_open_values_normalised_test, -1)
 
 
# print(len(test_data) - history_points)
 
 
next_day_open_values_test = np.array([test_data['Open'][2643 + i + history_points].copy() for i in range(len(test_data) - history_points)])
 
next_day_open_values_test = np.expand_dims(next_day_open_values_test, -1)
 
assert ohlcv_histories_normalised_test.shape[0] == next_day_open_values_normalised_test.shape[0]
 
ohlcv_train = ohlcv_histories_normalised
y_train = next_day_open_values_normalised
 
ohlcv_test =  ohlcv_histories_normalised_test
y_test = next_day_open_values_normalised_test
 
unscaled_y_test = next_day_open_values_test
 
number = int(0.9 * len(train_data))
# print(train_data[:number])
# print(len(ohlcv_train))
 
exponential_moving_average = [] #factors in more weight to recent data rather than a simple moving average which weighs everything equally (not good for stock data )
 
multiplier = 2.0 / 51
 
sma = np.mean(ohlcv_train[0][:,3])
 
exponential_moving_average.append(np.array([sma]))
 
for i in range(1, len(ohlcv_train)):
    sma = (train_data['Close'][i] - sma) * multiplier + sma
    exponential_moving_average.append(np.array([sma]))
 
scaler = preprocessing.MinMaxScaler()
exponential_moving_average_normalised = scaler.fit_transform(exponential_moving_average)
 
assert exponential_moving_average_normalised.shape[0] == y_train.shape[0]
 
 
exponential_moving_test = []
 
 
sma = np.mean(ohlcv_test[0][:,3])
 
exponential_moving_test.append(np.array([sma]))
 
for i in range(1, len(ohlcv_test)):
    sma = (test_data['Close'][i + 2643] - sma) * multiplier + sma
    exponential_moving_test.append(np.array([sma]))
 
exponential_moving_test_normalised = scaler.transform(exponential_moving_test)
 
assert exponential_moving_test_normalised.shape[0] == y_test.shape[0]
 
# print(exponential_moving_test_normalised)
 
# # print(ohlcv_histories_normalised.shape[0])
# # print(ohlcv_histories_normalised.shape[1])
# # print(ohlcv_histories_normalised.shape[2])
# # print(ohlcv_histories_normalised.shape[3])
# # print(ohlcv_histories_normalised.shape[4])
 
 
y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(next_day_open_values)
 
# print(exponential_moving_average_normalised.shape[1])
 
# def simple_model():
#     import keras
#     import tensorflow as tf
#     from keras.models import Model
#     from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
#     from keras import optimizers
#     from keras.callbacks import History 
#     import numpy as np
#     np.random.seed(4)
#     tf.random.set_seed(4)
 
#     # history = History()
 
 
#     lstm_input = Input(shape=(history_points, 6), name='lstm_input')
#     x = LSTM(50, name='lstm_0')(lstm_input)
#     x = Dropout(0.2, name='lstm_dropout_0')(x)
#     x = Dense(64, name='dense_0')(x)
#     x = Activation('sigmoid', name='sigmoid_0')(x)
#     x = Dense(1, name='dense_1')(x)
#     # x = Dropout(0.2, name='lstm_dropout_1')(x)
#     output = Activation('linear', name='linear_output')(x)
#     model = Model(inputs=lstm_input, outputs=output)
 
#     adam = optimizers.Adam(lr=0.0005)
 
#     model.compile(optimizer=adam, loss='mse')
 
#     return model
 
    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file='model.png')
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_iris
# from sklearn.ensemble import AdaBoostClassifier
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.ensemble import AdaBoostRegressor
 
# # estimator = KerasRegressor(build_fn = simple_model, epochs = 50, batch_size = 32, verbose = False)
 
# # boosted_result = AdaBoostRegressor(base_estimator = estimator)
# # boosted_result.fit(ohlcv_train, y_train)
 
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.callbacks import History 
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
 
# history = History()
 
 
lstm_input = Input(shape=(history_points, 6), name='lstm_input')
exponential_input = Input(shape=(exponential_moving_average_normalised.shape[1],), name='ema_input')
 
# x = LSTM(50, name='lstm_0')(lstm_input)
# x = Dense(70, name='dense_0')(x)
# # x = Dropout(0.2, name='lstm_dropout_0')(x)
# x = Activation('sigmoid', name='relu_1')(x)
# x = Dense(1, name='dense_2')(x)
 
# # x = Dropout(0.2, name='lstm_dropout_0')(x)
# # x = Dropout(0.2, name='lstm_dropout_1')(x)
# output = Activation('linear', name='linear_output')(x)
# model = Model(inputs=lstm_input, outputs=output)
 
# the first branch operates on the first input
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)
 
# the second branch opreates on the second input
y = Dense(32, name='avg_dense_0')(exponential_input)
y = Activation("relu", name='avg_relu_0')(y)
y = Dropout(0.2, name='avg_dropout_0')(y)
average_branch = Model(inputs=exponential_input, outputs=y)
 
 
# combine the output of the two branches
combined = concatenate([lstm_branch.output, average_branch.output], name='concatenate')
 
z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)
 
# our model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[lstm_branch.input, average_branch.input], outputs=z)
 
 
adam = optimizers.Adam(lr=0.0005)
 
model.compile(optimizer=adam, loss='mse')
 
model.fit(x=[ohlcv_train, exponential_moving_average_normalised], y=y_train, batch_size=32, epochs=100, shuffle=True, validation_split = 0.1) #callbacks = [history]
 
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)
 
#can pass in the validation state (validation_set)
 
#validation error
#evaluation metric (validation mse, train mse) #keras use plot history. 
# plot performance. every iteration has an mse value. to see whats happening.
#more layers. performance matric. 
 
#each iteration 200 jobs. train mse. run on the validation data. signs of overfitting on underfitting. trained mse goes down validation mse goes up 
#boosting LSTM article (BOOSTING, residual -> ) (boosting I )
# hmm -> HMM apply -> documentation. train HMM resources.  
#   
 
# model.save("my_model")
 
# model = keras.models.load_model('./my_model')
 
# losses = history.history
 
# evaluation = model.evaluate(ohlcv_train, y_train)
# print(evaluation)
# print(losses)
 
# plt.plot(losses['loss'][0:-1], label = 'Training Loss')
# plt.plot(losses['val_loss'][0:-1], label = 'Validation Loss')
 
# plt.legend(['Training Loss', 'Validation Loss'])
 
# plt.show()
 
# history.history.to_csv('./modelhistory.csv')
 
# histories = pd.read_csv('./modelhistory.csv')
 
# print(histories)
 
y_test_predicted = model.predict([ohlcv_test, exponential_moving_test_normalised])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
# y_predicted = model.predict(ohlcv_histories_normalised)
# y_predicted = y_normaliser.inverse_transform(y_predicted)
 
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)
 
plt.gcf().set_size_inches(22, 15, forward=True)
 
start = 0
end = -1
 
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
 
# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')
 
plt.legend(['Real', 'Predicted'])
 
plt.show()






# for i in range(len(stocks)):
#     print(stocks[i].isnull().sum().sum())
#no severe abnomalies. good! 

# for i in range(len(stocks)):
    # print(stocks[i][stocks[i]['High'] > stocks[i]['Open']])
    #check the high is higher than the open and close, and low is lower then the open and close

# for i in range(len(stocks)):
#     for j in range(0, 1000):
#         if j % 20 == 0:
#             print(j)
#             rand_number = randint(10, 20) #between 10-20 days of data
            






# dataset.append(['Adj Close'])
# print(panel_data.describe())
# test.append(panel_data)

# print(panel_data.isnull().sum().sum()) #check there is no null value inside. 
# print(test)
# test_stocks = pd.concat(test, axis = 1)

# df_stocks = pd.concat(panel_data, axis = 1)
# display = df_stocks.head(5)



# plt.figure(figsize=(16,8))
# plt.plot(df_stocks, label = "Adjusted close prize history")
# plt.show()



# print(eda(df_stocks))