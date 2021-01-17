import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

start_date = datetime(2010, 9, 1)
end_date = datetime(2020, 8, 31)

if __name__ == "__main__":

    stock_df = pd.read_csv('D:/didi/ml/stock-project/google_stocks_data.csv')


    new_df = pd.DataFrame({})

    new_df = stock_df[['Adj Close']].copy()

    train_split = 0.6

    new_df = new_df.loc[1:]

    train_split = 0.6

    data_set_points = 63

    train_indices = int(new_df.shape[0] * train_split)

    temps = new_df[train_indices:]
    temps = temps.reset_index()

    actual = np.array([temps['Adj Close'][i + data_set_points].copy() for i in range(len(temps) - data_set_points)])
    reference = temps['Adj Close'][data_set_points - 1]

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns = ['index'])

    train_data.head()


    train_arr = np.diff(train_data.loc[:, ['Adj Close']].values, axis = 0)
    test_arr = np.diff(test_data.loc[:, ['Adj Close']].values, axis = 0)

    print(train_arr)



    X_train = np.array([train_arr[i : i + data_set_points] for i in range(len(train_arr) - data_set_points)])

    y_train = np.array([train_arr[i + data_set_points] for i in range(len(train_arr) - data_set_points)])

    X_test = np.array([test_arr[i : i + data_set_points] for i in range(len(test_arr) - data_set_points)])

    y_test = np.array([new_df['Adj Close'][i + data_set_points] for i in range(len(test_arr) - data_set_points)])



    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print(X_test)


    import keras
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
    from keras import optimizers
    from keras.callbacks import History 
    import numpy as np
    np.random.seed(4)
    tf.random.set_seed(4)

    input_for_lstm = Input(shape=(data_set_points, 1), name='input_for_lstm')
    layer = LSTM(63, name='first_lstm_layer', return_sequences = True)(input_for_lstm)
    layer = Dropout(0.3, name='first_dropout_layer')(layer)
    layer = LSTM(32, name='lstm_1')(layer)
    layer = Dropout(0.2, name='lstm_dropout_1')(layer)
    layer = Dense(64, name='first_dense_layer')(layer)
    layer = Activation('sigmoid', name='sigmoid_0')(layer)
    layer = Dense(1, name='second_dense_layer')(layer)
    outputs= Activation('linear', name='linear_output')(layer)
    model = Model(inputs=input_for_lstm, outputs=outputs)

    adam = optimizers.Adam(lr = 0.004)

    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=56, epochs=1, shuffle=True, validation_split = 0.1) #callbacks = [history]

    y_pred = model.predict(X_test)

    print(y_pred)



    y_pred = y_pred.flatten()


    actual = np.array([test_data['Adj Close'][i + data_set_points].copy() for i in range(len(test_data) - data_set_points)])


    predicted = []
    reference = test_data['Adj Close'][data_set_points - 1]
    print(reference)
    predicted.append(reference)
    for i in y_pred:
        reference += i
        predicted.append(reference)
    predicted = np.array(predicted)
    print(predicted.size)
    print(actual.size)

    real = plt.plot(actual, label='Actual Price')
    pred = plt.plot(predicted, label='Predicted Price')



    plt.legend(['Actual Price', 'Predicted Price'])
    plt.gcf().set_size_inches(15, 10, forward=True)

    print(predicted)

    
    plt.show()
