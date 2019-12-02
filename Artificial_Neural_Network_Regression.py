# Start Python Imports
# Visualization
import matplotlib.pyplot as plt
from keras.callbacks import callbacks, ModelCheckpoint

from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Preprocessing
from sklearn.utils import shuffle
from keras import regularizers
# Neural Network
from keras.models import Sequential
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from keras import layers

# Preprocessing
from sklearn import preprocessing
# Evaluation
from sklearn.metrics import r2_score
# Ignore warnings
import warnings

warnings.filterwarnings('ignore')


class MLP:

    def __init__(self, window_size, optimizer, number_of_nodes):
        self.window_size = window_size
        self.optimizer = optimizer
        self.number_of_nodes = number_of_nodes
        self.patience = 100
        self.batch_size = 8
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.Dense(self.number_of_nodes, activation='relu', input_dim=self.window_size))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self, X_train, y_train, epochs, verbose=0):
        # fit model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, verbose=verbose,
                                      callbacks=callback_list)
        return self.history

    def predict(self, X_test):
        # demonstrate prediction
        X_test = X_test.reshape((1, self.window_size))
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=2)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)[1]

        residuals = y_test.T - prediction.T

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        return MSE, R2, residuals, good, bad


class LSTM:

    def __init__(self, window_size, no_of_features, optimizer, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        self.optimizer = optimizer
        self.number_of_nodes = number_of_nodes
        self.regularization_penalty = 0.0001
        self.patience = 100
        self.batch_size = 8
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.LSTM(self.number_of_nodes, input_shape=(self.window_size, self.no_of_features)))
        self.model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(self.regularization_penalty)))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self, X_train, X_test, y_train, y_test, number_of_epochs, verbose):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=number_of_epochs,
                                      verbose=verbose, validation_data=(X_test, y_test), callbacks=callback_list)

    def predict(self, X_test):
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=2)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)

        residuals = y_test.T - prediction.T

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        return MSE, R2, residuals, good, bad


class CNN:

    def __init__(self, window_size, no_of_features, optimizer, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        self.optimizer = optimizer
        self.number_of_nodes = number_of_nodes
        self.regularization_penalty = 0.0001
        self.patience = 100
        self.batch_size = 8
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.LSTM(self.number_of_nodes, input_shape=(self.window_size, self.no_of_features)))
        self.model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(self.regularization_penalty)))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self, X_train, y_train, epochs, verbose=0):
        # fit model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, verbose=verbose,
                                      callbacks=callback_list)

    def predict(self, X_test):
        # demonstrate prediction
        X_test = X_test.reshape((1, self.window_size))
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=2)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)[1]

        residuals = y_test.T - prediction.T

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        return MSE, R2, residuals, good, bad


class CNN_LSTM:

    def __init__(self, window_size, no_of_features, optimizer, number_of_filters_CNN, no_of_nodes_LSTM):
        self.window_size = window_size
        self.no_of_features = no_of_features
        self.optimizer = optimizer
        self.number_of_filters_CNN = number_of_filters_CNN
        self.no_of_nodes_LSTM = no_of_nodes_LSTM
        self.regularization_penalty = 0.0001
        self.patience = 100
        self.batch_size = 8
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.TimeDistributed(layers.Conv1D(filters=self.number_of_filters_CNN, kernel_size=1,
                                                            activation='relu'), input_shape=(None, self.window_size,
                                                                                             self.no_of_features)))
        self.model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.LSTM(self.no_of_nodes_LSTM, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self, X_train, X_test, y_train, y_test, epochs, verbose=0):
        X_train = X_train.reshape((X_train.shape[0], 1, 8, 2))
        X_test = X_test.reshape((X_test.shape[0], 1, 8, 2))
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, verbose=verbose,
                                      validation_data=(X_test, y_test), callbacks=callback_list)

    def predict(self, X_test):
        X_test = X_test.reshape((1, self.window_size))
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=2)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)[1]

        residuals = y_test.T - prediction.T

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        return MSE, R2, residuals, good, bad

    #
    # def ANN_regress(data, number_of_nodes, dropout_rate, window_size, batch_size, number_of_epochs,
    #                 regularization_penalty):
    #     """
    #     :param X: array of independent variables
    #     :param y: array of dependent variable
    #     :param number_of_nodes: number of neurons
    #     :param number_of_layers: number of layers
    #     :param dropout_rate: fraction rate between 0 and 1 of the input units to drop
    #     :param number_of_epochs: number of training iterations
    #     :param regularization_penalty: regularization penalty on layer parameters during optimization
    #     :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    #     :param sensor_name: name of the sensor user for plots
    #     :param score_name: name of the satisfaction score used for plots
    #     """
    #
    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #     data = data.to_numpy()
    #     no_features = data.shape[1] - 1
    #     data[:, 0:no_features] = scaler.fit_transform(data[:, 0:no_features])
    #
    #     def prepare_data(data, window_size):
    #         batches = []
    #         labels = []
    #         for idx in range(len(data) - window_size - 1):
    #             batches.append(data[idx: idx + window_size, 0:no_features])
    #             labels.append(data[idx + window_size, no_features])
    #         return np.array(batches), np.array(labels)
    #
    #     batches, labels = prepare_data(data, window_size)
    #
    #     X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42,
    #                                                         shuffle=True)
    #
    #     # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    #     # X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    #
    #     # Create a model with 2 hidden layers and 32 neurons using Rectified Linear Unit Activation Function
    #     # Add regularization (L2 - norm) and dropout to fight overfitting
    #
    #     model = Sequential()
    #
    #     # model.add(layers.LSTM(number_of_nodes, input_shape=(window_size, batches.shape[2])))
    #     # model.add(layers.Dropout(dropout_rate))
    #     # model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(regularization_penalty)))
    #
    #     # model.add(layers.Conv1D(filters=32, kernel_size=2, activation='relu', batch_input_shape=(None, batches.shape[1], batches.shape[2])))
    #     # model.add(layers.MaxPooling1D(pool_size=2))
    #     # model.add(layers.Flatten())
    #     # model.add(layers.Dense(50, activation='relu'))
    #     # model.add(layers.Dense(1))
    #
    #     model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
    #                                      input_shape=(None, 8, 2)))
    #     model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    #     model.add(layers.TimeDistributed(layers.Flatten()))
    #     model.add(layers.LSTM(100, activation='relu'))
    #     model.add(layers.Dense(1))
    #
    #     X_train = X_train.reshape((X_train.shape[0], 1, 8, 2))
    #     X_test = X_test.reshape((X_test.shape[0], 1, 8, 2))
    #
    #     model.compile(optimizer=opt,
    #                   loss='mean_squared_error',
    #                   metrics=['mean_squared_error'])
    #
    #     # checkpoint
    #     # filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    #     # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #     # callbacks_list = [checkpoint]
    #
    #     stop_no_improvement = [callbacks.EarlyStopping(monitor='val_loss', patience=100)]
    #
    #     training_history = model.fit(X_train, y_train,
    #                                  batch_size=batch_size, epochs=number_of_epochs, validation_data=(X_test, y_test),
    #                                  callbacks=stop_no_improvement)
    #
    #     y_predicted = model.predict(X_test, batch_size=batch_size, verbose=2)
    #
    #     # from keras.utils import plot_model
    #     # plot_model(model, to_file='model.png')
    #
    #     MSE_ANN = model.evaluate(X_test, y_test)[1]
    #
    #     residuals = y_test.T - y_predicted.T
    #
    #     R2_ANN = r2_score(y_test, y_predicted)
    #
    #     y_predicted = np.round(y_predicted, 0)
    #     fitness = y_predicted.T == y_test.T
    #     good = np.sum(fitness)
    #     bad = fitness.shape[1] - np.sum(fitness)
    #
    #     plt.figure(figsize=(12, 10))
    #     # Plot original data
    #     plt.scatter(y_predicted, residuals, color='red')
    #     plt.xlabel('predicted value')
    #     plt.ylabel('residual')
    #
    #     plt.figure(figsize=(12, 10))
    #     # Plot original data
    #     plt.scatter(y_predicted, y_test, color='blue')
    #     plt.xlabel('predicted value')
    #     plt.ylabel('real value')
    #
    #     plt.show()
    #
    #     return R2_ANN, MSE_ANN, y_predicted, training_history
