from trade_platform.src.agent.agent_thread import agent_thread
from ...src.util.util import *
import numpy as np
import matplotlib.pyplot as plt
from ...src.util.mrkt_data import mrkt_data
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
warnings.simplefilter('ignore', ConvergenceWarning)
#From https://github.com/philipperemy/keras-tcn
from tcn import TCN


def wave_net_activation(x): #https://www.kaggle.com/christofhenkel/temporal-cnn
    # type: (Layer) -> Layer
    """This method defines the activation used for WaveNet
    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/
    Args:
        x: The layer we want to apply the activation to
    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])

'''This was never used '''
#Loss function for punishing wrong decision of buy or sell and magnitude
def custom_loss(y, y_hat):
     mask = tf.math.multiply(y, y_hat)
     #print(mask)
     mask = tf.cast(mask < 0, mask.dtype) * mask
     mask = tf.math.abs(mask)
     yy = tf.math.multiply(y, mask)
     yy_hat = tf.math.multiply(y_hat, mask)
     #return backend.mean(backend.square(yy - yy_hat))
     return (backend.mean(backend.square(yy - yy_hat)) + backend.mean(backend.square(y - y_hat)))/2

class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at
    '''
    #########
    #add the following to train on the same dataset:
    #t = trade_platform(length=5000, data_path='data/US1.ABT_190504_200519.txt', enable_plot=False,random=False, type = "minute")
    #t.add_agent(tcn_agent(model = '1'))
    #t.start()
    #Can add tcn_agent with different perameters than default but changing below. setting arima to True automatically add
    #one to the input dimensions to account for the extra feature
    #Specify the model you want to run under
    #########
    #add the following to train on different dataset:
    #t = trade_platform(length=5000, data_path='data/US1.ABT_190504_200519.txt', enable_plot=False, random=False,
    #                   type="minute")
    #trained_agent = tcn_agent(model = '1', trainset = 0, moments = 25, arima = True)
    # trained_agent.train('data/US1.ATVI_200505_200507.txt')
    #t.add_agent(trained_agent)
    #t.start()
    #Can change perameters, just need to set trainset to 0 when running on another dataset

    def __init__(self, moments = 17, batch_size = None, input_dim = 6, model = '1', trainset = 100, arima = True, loadModel = False, reproduceable = False, norm = "custom"):
        if reproduceable:
            np.random.seed(2020)
        agent_thread.__init__(self)
        self.train_dif = 0
        self.moments = moments #Number of moments looked back to decide next value
        self.holding_time = 0
        self.batch_size = batch_size
        self.input_dim = input_dim #Dimensions of input default to 1
        self.arima_on = arima  # Neither to use arima or not, show run and training time but increase accuracy
        if self.arima_on == True:
            self.input_dim += 1
        self.built = False
        if loadModel:
            self.m = load_model(loadModel, custom_objects={'TCN': TCN})
            self.stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
            self.m.compile(optimizer='adam', loss='mse')
            if model == '1':
                self.save = ModelCheckpoint(f'./model1{norm}.h5', save_best_only=True, monitor='val_loss', mode='min')
            elif model == '2':
                self.save = ModelCheckpoint('./model2.h5', save_best_only=True, monitor='val_loss', mode='min')
            elif model == '3':
                self.save = ModelCheckpoint('./model3.h5', save_best_only=True, monitor='val_loss', mode='min')
            elif model == '4':
                self.save = ModelCheckpoint('./model4.h5', save_best_only=True, monitor='val_loss', mode='min')
        else:
            self.model(model) #Compile model
        self.training = True
        self.networth = 0
        self.amount = 0 #stocks bought
        self.correct_guess = 0 #tracker to see when guessing correct of open prices
        self.sergei = 0 #Tracks how much sergei's algoritm gains or losses
        self.arima = []
        self.features = []
        self.scaler = 0
        self.buy_points = []
        self.sell_points = []
        self.networth_points = []
        self.training_data = moments*trainset #Amount of data to set aside for training when running on same data set

    def _find_decision(self):
        #Right now these below if statement are for training and testing on the same data set with training on
        #the first self.moments*10 and testing on the remaining
        offset = 0
        if self.arima_on:
            offset = 50
        if len(self.market_history)%500 == 0 and len(self.market_history) > self.training_data + offset:
            self.plot()
        if self.training_data == 0:
            pass
        elif len(self.market_history) == self.training_data + offset:
            self.train()
        if len(self.market_history) > self.training_data + offset + self.moments:
            predicted_value, real_current_value = self.run_model()
            print("Correct guess chance is: ", self.correct_guess*100/(self.time_counter - self.training_data - offset - self.moments), "%")
            if not self.holding and predicted_value[0] > 0: #Change to real_current_value[0] if using scaler
                self.amount = 100 #amount to buy is set to fix for now but can be changed
                #self.amount = 1000 * (predicted_value[0] - real_current_value[0]) use this for varing amount or make your own
                if self.amount < 0:
                    self.amount = 1
                self.act = action.BUY
                self.holding = True
                self.holding_time = self.time_counter
                ###Sergei's algorithm
                # looking at self.time_counter-2 so we can check the real value while guessing the real value
                # as well as be able to use high and low of current market point for Sergei's algorithm
                #if (self.market_history[self.time_counter - 3].close >= self.market_history[self.time_counter - 3].open
                #        and self.market_history[self.time_counter - 2].close <= self.market_history[self.time_counter - 2].open
                #        and (self.market_history[self.time_counter - 3].high >= self.market_history[self.time_counter - 1].low
                #        and self.market_history[self.time_counter - 3].high <= self.market_history[self.time_counter - 1].high)):
                #    self.buy_in_price = self.market_history[self.time_counter - 3].high
                #    print("Used Sergei's algorithm and gained/lost ", self.market_history[self.time_counter - 2].price - self.buy_in_price, " per share")
                #    self.sergei += (self.market_history[self.time_counter - 2].price - self.buy_in_price) * self.amount
                #else:
                self.buy_in_price = self.market_history[self.time_counter - 2].price #base price
                #print(self.buy_in_price)
                self.buy_points.append([self.time_counter-1, self.buy_in_price])
                self.networth -= self.buy_in_price * self.amount
                print("buy  at time " + str(self.time_counter) + "\t price : " + str(self.buy_in_price))
            elif self.holding and predicted_value[0] < 0:#Change to real_current_value[0] if using scaler
                self.act = action.SELL
                ###Sergei's algorithm
                #if (self.market_history[self.time_counter - 3].close <= self.market_history[self.time_counter - 3].open
                #        and self.market_history[self.time_counter - 2].close >= self.market_history[self.time_counter - 2].open
                #        and (self.market_history[self.time_counter - 3].low >= self.market_history[self.time_counter - 1].low
                #        and self.market_history[self.time_counter - 3].low <= self.market_history[self.time_counter - 1].high)):
                #    self.sell_price = self.market_history[self.time_counter - 3].low
                #    print("Used Sergei's algorithm and gained/lost ", self.buy_in_price - self.market_history[self.time_counter - 2].price, " per share")
                #    self.sergei += (self.buy_in_price - self.market_history[self.time_counter - 2].price) * self.amount
                #else:
                self.sell_price = self.market_history[self.time_counter - 2].price
                self.sell_points.append([self.time_counter - 1, self.sell_price])
                self.networth += self.sell_price * self.amount#base price
                self.networth_points.append([self.time_counter -1, self.networth])
                #print(self.market_history[self.time_counter - 2].price) #selling price base
                self.amount = 0 #reset amount to account for varying amounts
                print("sell at time " + str(self.time_counter) + "\t price : " + str(
                    self.sell_price))
                print("Current networth is: ", self.networth)
                print("Sergei's networth: ", self.sergei)
                self.holding = False
            elif self.holding and self.time_counter - self.holding_time > 20:
                self.act = action.SELL
                self.networth += self.market_history[self.time_counter - 2].price * self.amount
                print("hold 2 long  " + str(self.time_counter) + "\t price : " + str(
                    self.market_history[self.time_counter - 1].price))
                self.holding = False
            else:
                print(self.time_counter - self.holding_time)
                self.act = action.HOLD
            return self.act
        else:
            pass

    def model(self, model):
        #build model around TCN
        #As a general rule (keeping kernel_size fixed at 2) and dilations increasing with a factor of 2
        #The equation to find the ideal sizes is receptive field = nb_stacks_of_residuals_blocks(nb_stacks) * kernel_size * last_dilation)
        #Each layer adds linearly to the receptive field
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        #Add models below to test and report back, need a way to save models as well as graphing results
        #All need optimizing
        if model == '1': #report results, I tried this model with the default settings on the same dataset with
            #the first 1750 data points for training and arima set to true. Recieved a 58% accuracy over 200 data points
            #and increased networth, slight loss of income due to sergei's algoritm
            #Can try with activation = wave_net_activation
            x1 = TCN(return_sequences=False, nb_filters=(self.moments-1)*2, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.3,
                     kernel_size=2)(i)
            x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(i)
            x2 = TCN(return_sequences=False, nb_filters=(self.moments-1)*2, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.1,
                     kernel_size=2)(x2)
            x = add([x1, x2])
            o = Dense(1, activation='linear')(x)
            self.save = ModelCheckpoint('./model1.h5', save_best_only=False, monitor='val_loss', mode='min')

        elif model == '2':
            x1 = TCN(return_sequences=True, nb_filters=(self.moments - 1) * 2, dilations=[1, 2, 4], nb_stacks=2,
                     dropout_rate=.3,
                     kernel_size=2)(i)
            x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(i)
            x2 = TCN(return_sequences=True, nb_filters=(self.moments - 1) * 2, dilations=[1, 2, 4], nb_stacks=2,
                     dropout_rate=.1,
                     kernel_size=2)(x2)
            x = add([x1, x2])
            x1 = LSTM(5, return_sequences=False, dropout=.3)(x)
            x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(x)
            x2 = LSTM(5, return_sequences=False, dropout=.3)(x2)
            x = add([x1, x2])
            o = Dense(1, activation='linear')(x)
            self.save = ModelCheckpoint('./model2.h5', save_best_only=True, monitor='val_loss', mode='min')

        elif model == '3':
            #Complex model to test and change, probably poorer results due to overtraining
            x = TCN(return_sequences=True, nb_filters=32, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.3,
                     kernel_size=4)(i)
            x1 = TCN(return_sequences=True, nb_filters = 16, dilations = [1, 2, 4, 8], nb_stacks = 2, dropout_rate=.3, kernel_size=4)(x)
            x2 = LSTM(32, return_sequences=True, dropout=.3)(i)
            x2 = LSTM(16, return_sequences=True, dropout=.3)(x2)
            x = add([x1, x2])
            x = Dense(8, activation='linear')(x)
            x = TCN(return_sequences=True, nb_filters=4, dilations=[1, 2, 4], nb_stacks=1, dropout_rate=.3,
                    kernel_size=2, activation=wave_net_activation)(x)
            x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
            o = Dense(1, activation='linear')(x)
            self.save = ModelCheckpoint('./model3.h5', save_best_only=True, monitor='val_loss', mode='min')

        elif model == '4':
            x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16], nb_stacks = 1, dropout_rate=.1, kernel_size=4)(i)
            x1 = TCN(return_sequences=True, nb_filters = 16, dilations = [1, 2, 4, 8, 16], nb_stacks = 1, dropout_rate=.1, kernel_size=4)(x1)
            x1 = Dense(4, activation='linear')(x1)
            x2 = LSTM(4, dropout=.3)(i)
            x = add([x1, x2])
            o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
            o = Dense(1, activation='linear')(o)
            self.save = ModelCheckpoint('./model4.h5', save_best_only=True, monitor='val_loss', mode='min')
        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss=custom_loss) #optimizer and loss can be changed to what we want
        self.stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        #########################################################################

    def arima_feature(self, i):
        try:
            if len(self.arima) < 27:
                m = ARIMA(self.arima[:i], order=(5, 1, 0))
            else:
                m = ARIMA(self.arima[i-27:i], order=(5,1,0))
            m_fit = m.fit(disp=0)
            output = m_fit.forecast()
            print(i)
            #print(output[0][0])
            return output[0][0]
        except:
            return self.arima[i]

    def get_technical_indicators(self, i):
        if len(self.market_history) >= 8 and i >= 8:
            last7 = self.market_history[i-8:i-1]
        else:
            last7 = self.market_history[:]
        if len(self.market_history) >= 28 and i >= 28:
            last21 =  self.market_history[i-28:i-1]
        else:
            last21 = self.market_history[:]
        last7 = [i.price for i in last7]
        last21 = [i.price for i in last21]
        malast7 = np.mean(last7)
        malast21 = np.mean(last21)

        return malast7, malast21

    def get_technical_indicators_(self, i, input):
        if len(input) >= 8 and i >= 8:
            last7 = input[i-8:i-1]
        else:
            last7 = input[:]
        if len(input) >= 28 and i >= 28:
            last21 = input[i-28:i-1]
        else:
            last21 = input[:]
        last7 = [i[0] for i in last7]
        last21 = [i[0] for i in last21]
        malast7 = np.mean(last7)
        malast21 = np.mean(last21)

        return malast7, malast21

    def split_data(self, input, moments, lookahead = 1):
        print("splitting data")
        # Split data into groups for training and testing
        size = len(input)
        if self.training_data == 0 and len(self.features) == 0:
            self.prepare_data_(input)
        else:
            self.prepare_data()
        input = self.features[-size:]
        input = np.array(input)
        input = np.atleast_2d(input)
        x = np.array([])
        y = np.array([])
        #Normalize data
        input = self.normalization(input, 'custom')
        for i in range(input.shape[0] - moments+1):
            x_values = np.array(input[i:moments + i - lookahead])
            y_values = np.array(input[i+moments-lookahead:i+moments][0][0])
            if (x.shape[0] == 0):
                x = x_values
                if len(y_values.shape) == 1:
                    y = [y_values]
                else:
                    y = y_values
            else:
                if i == 1:
                    x = np.concatenate(([x], [x_values]))
                    y = np.concatenate(([y], [y_values]))
                else:
                    x = np.concatenate((x, [x_values]))
                    y = np.concatenate((y, [y_values]))
        if len(x.shape) < 3:
            x = x.reshape((1, x.shape[0], x.shape[1]))
        return x, y

    def normalization(self, data, mode = 'default'):
        #To be added to with normalization methods
        #Time consuming can be improved
        if mode == 'default':
            if self.scaler == 0:
                self.scaler = preprocessing.StandardScaler()
                return self.scaler.fit_transform( data )
            return self.scaler.transform( data )
            #if self.scaler == 0:
            #    self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            #    self.scaler = self.scaler.fit(data)
            #    return self.scaler.fit_transform(data)
            #return self.scaler.transform(data)
        elif mode == 'lognormal':
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        inner_data.append(np.log(value / value))
                    else:
                        inner_data.append(np.log(value / data[i-1][j]))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == 'custom':
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        if (j < 1):
                            inner_data.append(np.log(value / value))
                        else:
                            inner_data.append(np.log(value/data_pt[0]))
                    else:
                        if (j < 1):
                            inner_data.append(np.log(value / data[i - 1][0]))
                        else:
                            inner_data.append(np.log(value/ (data[i - 1][0])))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == "percentile":
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        if (j < 1):
                            inner_data.append((value / value)-1)
                        else:
                            inner_data.append((value / data_pt[j])-1)
                    else:
                        if (j < 1):
                            inner_data.append((value / data[i - 1][j])-1)
                        else:
                            inner_data.append((value / data[i-1][j])-1)#data[i - 1][0])-1)
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        return data

    def prepare_data(self):
        difference = len(self.features) - len(self.market_history) - self.train_dif
        if self.arima_on:
            difference +=50
        if difference < 0:
            input = [[i.open, i.close, i.low, i.high] for i in self.market_history[difference:]]
            # input = [[i.open, i.close, (i.high-i.close+offset)*2/(i.close+i.high)] for i in input]
            # input = [[i.open, i.close] for i in input]
            for i in range(len(input)):
                if self.arima_on == True:
                    input[i].append(self.arima_feature(len(self.arima) - len(input) + i))
                input[i].append(self.get_technical_indicators(len(self.arima) - len(input) + i)[0])
                input[i].append(self.get_technical_indicators(len(self.arima) - len(input) + i)[1])
                self.features.append(input[i])

    def prepare_data_(self, input):
        self.train_dif = len(input)
        difference = len(self.features) - len(input)
        if self.arima_on:
            difference +=50
        if difference < 0:
            input = [[i.open, i.close, i.low, i.high] for i in input[difference:]]
            # input = [[i.open, i.close, (i.high-i.close+offset)*2/(i.close+i.high)] for i in input]
            # input = [[i.open, i.close] for i in input]
            for i in range(len(input)):
                if self.arima_on == True:
                    input[i].append(self.arima_feature(len(self.arima) - len(input) + i))
                input[i].append(self.get_technical_indicators_(len(self.arima) - len(input) + i, input)[0])
                input[i].append(self.get_technical_indicators_(len(self.arima) - len(input) + i, input)[1])
                self.features.append(input[i])

    def train(self, data_path = None):
        def read_data():
            if data_path == None:
                return data_path
            training_data = list()
            values = parse(data_path).values
            for i, val in enumerate(values):
                training_data.append(mrkt_data(val, time=i));
            return training_data
        inputs = read_data()
        if inputs == None:
            if self.arima_on:
                inputs = self.market_history[50:]
            else:
                inputs = self.market_history[:]
        x, y = self.split_data(inputs, self.moments)
        self.m.fit(x, y, epochs=500, validation_split=0.1, callbacks=[self.stop, self.save])

    def run_model(self):
        #if self.log_percentage != []:
            #inputs = self.get_log_percentages(-self.moments)
        #else:
        inputs = self.market_history[-self.moments:]

        x, y = self.split_data(inputs, self.moments)
        #print(x)
        #print(y)
        y_hat = self.m.predict(x)
        print("Predicting next price to be: ", y_hat[0][0])
        print("Real next price was: ", y)
        #if ((y-x[0][-1][0]) * (y_hat[0][0]-x[0][-1][0]) > 0): #use if scaler
        if (y * y_hat[0][0] > 0): #Use if percentile
            self.correct_guess +=1
        y_normal = 0
        if self.percentage != []:
            y_normal = (y_hat[0][0])*self.market_history[-2].price
        else:
            y_normal = 10*(y_hat[0][0]/self.market_history[-2].price)*self.market_history[-2].price
        #print("Converted back predicted value is ", y_normal) #To be changed to function to account for different normalized data
        #print("Read normal value is ", self.market_history[-1].price)
        return y_hat[0], x[0][-1]

    def update_market_history(self, data):
        # undate for 1 unit of time
        self.next_time()
        self.market_history.append(data)
        self.arima.append(data.price)

    def plot(self):
        plt.plot(np.arange(len(self.market_history)-499, len(self.market_history)), [self.features[i][0] for i in np.arange(len(self.features)-499, len(self.features))], label="market")
        offset = 0
        if self.arima_on:
            offset = 50
        if len(self.market_history) > self.training_data + offset + 50:
            plt.scatter([i[0] for i in self.buy_points[:] if i[0] > self.time_counter - 500], [i[1] for i in self.buy_points[:] if i[0] > self.time_counter - 500], label="buy")
            plt.scatter([i[0] for i in self.sell_points[:] if i[0] > self.time_counter - 500], [i[1] for i in self.sell_points[:] if i[0] > self.time_counter - 500], label="sell")
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        plt.show()
        if len(self.market_history) > self.training_data + offset + 50:
            plt.plot([i[0] for i in self.networth_points[:] if i[0] > self.time_counter - 500], [i[1] for i in self.networth_points[:] if i[0] > self.time_counter - 500], label="networth")
            plt.xlabel('time')
            plt.ylabel('dollars')
            plt.legend()
            plt.show()

    def set_exit(self, value=True):
        # used to exit this agent thread
        self.exit = value
        #print("Sergei's networth: ", self.sergei) #If using sergei's algorithm
        print("Networth: ", self.networth)
