import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import regularizers
from keras.callbacks import callbacks
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings('ignore')


class MLP:

    def __init__(self, window_size, no_of_features, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.number_of_nodes = number_of_nodes
        self.patience = 500
        self.batch_size = 8
        self.model = Sequential()
        self.model.add(layers.Dense(self.number_of_nodes, activation='relu', input_shape=(self.window_size,
                                                                                          self.no_of_features)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

    def predict(self, X_test):
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

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')

        plt.show()


class LSTM:

    def __init__(self, window_size, no_of_features, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.number_of_nodes = number_of_nodes
        self.regularization_penalty = 0.
        self.patience = 500
        self.batch_size = 8
        self.model = Sequential()
        self.model.add(layers.LSTM(self.number_of_nodes, input_shape=(self.window_size, self.no_of_features)))
        self.model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(self.regularization_penalty)))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=no_of_epochs,
                                 verbose=verbose, validation_data=(X_test, y_test), callbacks=callback_list)
        return history

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

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')

        plt.show()


class CNN:

    def __init__(self, window_size, no_of_features, number_of_filters):
        self.window_size = window_size
        self.no_of_features = no_of_features
        self.number_of_filters = number_of_filters
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.regularization_penalty = 0.
        self.patience = 50
        self.batch_size = 8
        self.model = Sequential()
        self.model.add(layers.Conv1D(filters=number_of_filters, kernel_size=2, activation='relu',
                                     batch_input_shape=(None, self.window_size, self.no_of_features)))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(50, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

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

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')

        plt.show()


class CNN_LSTM:

    def __init__(self, window_size, no_of_features, number_of_filters_CNN, no_of_nodes_LSTM):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.number_of_filters_CNN = number_of_filters_CNN
        self.no_of_nodes_LSTM = no_of_nodes_LSTM
        self.regularization_penalty = 0.
        self.patience = 500
        self.batch_size = 8
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.TimeDistributed(layers.Conv1D(filters=self.number_of_filters_CNN, kernel_size=1,
                                                            activation='relu'), input_shape=(None, self.window_size,
                                                                                             self.no_of_features)))
        self.model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.LSTM(self.no_of_nodes_LSTM))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, verbose=1):
        X_train = X_train.reshape((X_train.shape[0], 1, 8, 2))
        X_test = X_test.reshape((X_test.shape[0], 1, 8, 2))
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

    def predict(self, X_test):
        X_test = X_test.reshape((X_test.shape[0], 1, 8, 2))
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=2)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        X_test = X_test.reshape((X_test.shape[0], 1, 8, 2))
        MSE = self.model.evaluate(X_test, y_test)

        residuals = y_test.T - prediction.T

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        return MSE, R2, residuals, good, bad

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')

        plt.show()
