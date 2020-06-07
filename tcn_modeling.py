from trade_platform.src.agent.agent_thread import agent_thread
from trade_platform.src.util.util import *
import numpy as np
import matplotlib.pyplot as plt
from trade_platform.src.util.mrkt_data import mrkt_data
from tensorflow.keras.layers import Dense, add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate, LSTM, Activation, multiply
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from trade_platform.src.util.Data_parsing.data_parsing import parse
import tensorflow as tf
from statsmodels.tsa.arima_model import ARIMA
import warnings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tcn import TCN

class tcn():

    def __init__(self, moments, model = 1, data_path = None, batch_size = None, input_dims = 6, trainset = 100, loadModel = None, reporducability = False):
        if reporducability:
            np.random.seed(2020)

        self.batch_size = batch_size

        save = f"model_{model}+moments_{moments}+batch_size{batch_size}.h5"
        self.save = ModelCheckpoint(save, save_best_only=True, monitor='val_loss', mode='min')
        self.moments = moments;
        self.stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        self.x, self.y = self.get_data(data_path)

        if loadModel == None:
            '''make the model here'''
            i = Input(batch_shape=(self.batch_size, self.moments, 4))
            if model == 1:
                x1 = TCN(return_sequences=False, nb_filters=(self.moments)*2, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.3,
                         kernel_size=2)(i)
                x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(i)
                x2 = TCN(return_sequences=False, nb_filters=(self.moments)*2, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.1,
                         kernel_size=2)(x2)
                x = add([x1, x2])
                o = Dense(1, activation='linear')(x)

            elif model == 2:
                x1 = TCN(return_sequences=True, nb_filters=(self.moments) * 2, dilations=[1, 2, 4], nb_stacks=2,
                     dropout_rate=.3,
                     kernel_size=2)(i)
                x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(i)
                x2 = TCN(return_sequences=True, nb_filters=(self.moments) * 2, dilations=[1, 2, 4], nb_stacks=2,
                         dropout_rate=.1,
                         kernel_size=2)(x2)
                x = add([x1, x2])
                x1 = LSTM(5, return_sequences=False, dropout=.3)(x)
                x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(x)
                x2 = LSTM(5, return_sequences=False, dropout=.3)(x2)
                x = add([x1, x2])
                o = Dense(1, activation='linear')(x)

            self.m = Model(inputs=i, outputs=o)

        else:
            self.m = load_model(loadModel, custom_objects = {'TCN': TCN})

        self.m.summary();
        self.m.compile(optimizer='adam', loss='mse')

    def test_set(self, data_path = None):
        # takes data from a path and uses as test set
        self.x_test, self.y_test = self.get_data(data_path)

    def get_data(self, data_path = None):
        # takes a data path and from there returns the properly split data
        # returns the proper shape and the expected y
        if data_path != None:
            values = parse(data_path).values
        else:
            print("No data passed")
            return

        self.data = list()
        for i, val in enumerate(values):
            self.data.append(val)

        self.data = np.asarray(self.data)

        # noramlize the data
        self.data = self.normalized()
        return self.split_data()

    def split_data(self):
        # x values are the moments that the network gets to see
        x = np.asarray([self.data[i: i+ self.moments] for i in range(len(self.data)-self.moments)])

        # y values are the moments after
        y = np.asarray([self.data[i+ self.moments][0] for i in range(len(self.data)-self.moments)])
        # print(x[0], y[0])

        return x,y

    def normalized(self):
        #log percentage normalization
        normalized_data = list()
        for i, data_pt in enumerate(self.data):
            inner_data = list()
            for j, value in enumerate(data_pt):
                if (i == 0):
                    if (j < 1):
                        inner_data.append(np.log(value / value))
                    else:
                        inner_data.append(np.log(value/data_pt[0]))
                else:
                    if (j < 1):
                        inner_data.append(np.log(value / self.data[i - 1][0]))
                    else:
                        inner_data.append(np.log(value/ (self.data[i - 1][0])))
            normalized_data.append(inner_data)
        return np.array(normalized_data)

    def train(self):
        print(f'the number of data poitns is: {len(self.x)}')
        self.m.fit(self.x, self.y, batch_size = 32, epochs=500, validation_split=0.1, callbacks=[self.stop, self.save])

    def test(self):
        results = self.m.evaluate(self.x_test, self.y_test)
        print('test loss, test acc:', results)

    def predict(self):
        #return predictions and what the values should have been to compare
        predictions = self.m.predict(self.x_test)
        return predictions, self.y_test
