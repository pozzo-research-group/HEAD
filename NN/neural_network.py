# Import plotting libraries
import matplotlib 
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.size': 22})
#import keras
import tqdm
from tqdm.keras import TqdmCallback
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import regularizers

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class neural_net():
    def __init__(self):
        return
    
    def load_data(self):
        for i in range(10):
            x = pd.read_csv('../Data/Volumes/Itr' + str(i) + '.csv').values
            x = np.delete(x, 0, axis=1)
            x = np.delete(x, 1, axis=1)
            if i == 0:
                all_x = x
            else:
                all_x = np.vstack((all_x, x[0:-1,:]))
                
        for i in range(10):
            y = pd.read_excel('../Data/Spectra/Itr_' + str(i) + '.xlsx').values
            self.wavelength = y[:,0]
            y = np.delete(y, 0, axis = 1)
            if i == 0:
                all_y = y
            else:
                all_y = np.hstack((all_y, y[:, 0:-1]))
        #Normalization 
        minmax = MinMaxScaler()
        all_x = minmax.fit_transform(all_x)
        all_y = minmax.fit_transform(all_y)              
        x = all_x
        y = all_y 
        X_train, X_test, y_train, y_test = train_test_split(x, y.T,
                                                    test_size=0.05,
                                                    random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return 
    
    def medium_network(self):
        # assemble the structure
        model = Sequential()
        model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(101, kernel_initializer='normal'))
        #opt = tf.keras.optimizers.Adam(learning_rate=0.9)
        # compile the model
        model.compile(loss='mean_squared_error', optimizer='Adam')
        return model

    def train_model(self):
        # initialize the andom seed as this is used to generate
        # the starting weights
        X_train = self.X_train
        y_train = self.y_train
        # create the NN framework
        estimator = KerasRegressor(build_fn= self.medium_network,
                epochs=150, batch_size=10000, verbose=0)
        history = estimator.fit(X_train, y_train, validation_split=0.30, epochs=150, 
                batch_size=10000, verbose=0, callbacks=TqdmCallback(verbose=0))
        print("Final MSE for train is %.3e and for validation is %.3e" % 
              (history.history['loss'][-1], history.history['val_loss'][-1]))
        self.estimator = estimator
        return 

    def predict(self, x_values):
        minmax = MinMaxScaler()
        y_pred = self.estimator.predict(x_values)
        if x_values.shape[0] == 1:
            y_pred = minmax.fit_transform(y_pred.reshape(-1,1))
        else:
            y_pred = minmax.fit_transform(y_pred.T)
            y_pred = y_pred.T
        return y_pred, self.wavelength
    
    def plot_predictions(self):
        minmax = MinMaxScaler()
        self.y_pred = self.estimator.predict(self.X_test)
        self.y_pred = minmax.fit_transform(self.y_pred.T)
        self.y_pred = self.y_pred.T
        for i in range(self.y_pred.shape[0]):
            fig, ax = plt.subplots()
            plt.plot(self.wavelength, self.y_pred[i,:], color = 'red', label = 'Prediction')
            plt.plot(self.wavelength, self.y_test[i,:], color = 'black', label = 'Actual')
            plt.legend()
            
    
        
        
        
        
                