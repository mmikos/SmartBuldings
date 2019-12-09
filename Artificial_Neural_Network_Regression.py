import numpy as np
import matplotlib.pyplot as plt
from keras import layers, metrics, initializers
from keras import regularizers
from keras.callbacks import callbacks, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import r2_score

from keras import backend as K
import warnings

from tensorflow import keras

warnings.filterwarnings('ignore')


def coeff_determination(y_true, y_pred):

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    R2 = (1 - SS_res / (SS_tot + K.epsilon()))

    return R2


def root_mean_squared_error(y_true, y_pred):
    rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
    return rmse
# metrics.Precision()


class MLP:

    def __init__(self, window_size, no_of_features, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.number_of_nodes = number_of_nodes
        self.patience = 200
        self.batch_size = 128
        self.loss = root_mean_squared_error
        self.model = Sequential()
        self.model.add(layers.Dense(self.number_of_nodes, activation='relu', input_shape=(self.window_size,
                                                                                          self.no_of_features)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation='relu'))
        # self.model.load_weights("weights/weights.best.MLP.hdf5")
        self.model.compile(optimizer=opt, loss=self.loss)

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, batch_size, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        # filepath = "weights/weights.best.MLP.hdf5"
        # callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min'),
        #                  ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min'),
        #                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        #                  ]
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

    def predict(self, X_test):
        prediction = self.model.predict(X_test)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)

        residuals = np.abs(np.round(y_test.T - prediction.T, 0))

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        exact_accuracy = good / (good + bad)

        mean_error = np.mean(residuals)

        return MSE, R2, residuals, exact_accuracy, mean_error

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')
        plt.title('Results of MLP Network - Residuals')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')
        plt.title('Results of MLP Network - Real vs Predicted')
        plt.show()


class LSTM:

    def __init__(self, window_size, no_of_features, number_of_nodes):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.1, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.loss = root_mean_squared_error
        self.number_of_nodes = number_of_nodes
        self.regularization_penalty = 0.
        self.patience = 200
        self.batch_size = 128
        self.model = Sequential()
        self.model.add(layers.LSTM(self.number_of_nodes, input_shape=(self.window_size, self.no_of_features)))
        self.model.add(layers.Dense(1, activation='relu', kernel_regularizer=regularizers.l2(self.regularization_penalty)))
        # self.model.load_weights("weights/weights.best.LSTM.hdf5")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, batch_size, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        # filepath = "weights/weights.best.LSTM.hdf5"
        # callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min'),
        #                  ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min'),
        #                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        #                  ]
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=no_of_epochs,
                                 verbose=verbose, validation_data=(X_test, y_test), callbacks=callback_list)
        # , callbacks = callback_list
        return history

    def predict(self, X_test):
        prediction = self.model.predict(X_test)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)

        residuals = np.abs(np.round(y_test.T - prediction.T, 0))

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        exact_accuracy = good / (good + bad)

        mean_error = np.mean(residuals)

        return MSE, R2, residuals, exact_accuracy, mean_error

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')
        plt.title('Results of LSTM Network - Residuals')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')
        plt.title('Results of LSTM Network - Real vs Predicted')
        plt.show()


class CNN:

    def __init__(self, window_size, no_of_features, number_of_filters):
        self.window_size = window_size
        self.no_of_features = no_of_features
        self.number_of_filters = number_of_filters
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.loss = root_mean_squared_error
        self.regularization_penalty = 0.
        self.patience = 200
        self.batch_size = 128
        self.model = Sequential()
        self.model.add(layers.Conv1D(filters=number_of_filters, kernel_size=2, activation='relu',
                                     batch_input_shape=(None, self.window_size, self.no_of_features)))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(50))
        self.model.add(layers.Dense(1, activation='relu'))
        # self.model.load_weights("weights/weights.best.CNN.hdf5")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, batch_size, verbose=1):
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        # filepath = "weights/weights.best.CNN.hdf5"
        # callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min'),
        #                  ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min'),
        #                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience, verbose=1, epsilon=1e-4, mode='min')
        #                  ]
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

    def predict(self, X_test):
        prediction = self.model.predict(X_test)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        MSE = self.model.evaluate(X_test, y_test)

        residuals = np.abs(np.round(y_test.T - prediction.T, 0))

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        exact_accuracy = good / (good + bad)

        mean_error = np.mean(residuals)

        return MSE, R2, residuals, exact_accuracy, mean_error

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')
        plt.title('Results of CNN Network - Residuals')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')
        plt.title('Results of CNN Network - Real vs Predicted')
        plt.show()


class CNN_LSTM:

    def __init__(self, window_size, no_of_features, number_of_filters_CNN, no_of_nodes_LSTM):
        self.window_size = window_size
        self.no_of_features = no_of_features
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        self.optimizer = opt
        self.loss = root_mean_squared_error
        self.number_of_filters_CNN = number_of_filters_CNN
        self.no_of_nodes_LSTM = no_of_nodes_LSTM
        self.regularization_penalty = 0.
        self.patience = 200
        self.batch_size = 64
        self.training_history = None
        self.model = Sequential()
        self.model.add(layers.TimeDistributed(layers.Conv1D(filters=self.number_of_filters_CNN, kernel_size=1,
                                                            activation='relu'), input_shape=(None, self.window_size,
                                                                                             self.no_of_features)))
        self.model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.LSTM(self.no_of_nodes_LSTM))
        # self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation='relu'))
        # self.model.load_weights("weights/weights.best.CNN_LSTM.hdf5")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X_train, X_test, y_train, y_test, no_of_epochs, verbose=1):
        X_train = X_train.reshape((X_train.shape[0], 1, self.window_size, self.no_of_features))
        X_test = X_test.reshape((X_test.shape[0], 1, self.window_size, self.no_of_features))
        callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        # filepath = "weights/weights.best.CNN_LSTM.hdf5"
        # callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min'),
        #                  ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min'),
        #                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        #                  ]
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=no_of_epochs, verbose=verbose,
                                 validation_data=(X_test, y_test), callbacks=callback_list)
        return history

    def predict(self, X_test):
        X_test = X_test.reshape((X_test.shape[0], 1, self.window_size, self.no_of_features))
        prediction = self.model.predict(X_test, self.batch_size)
        return prediction

    def evaluate(self, prediction, X_test, y_test):
        X_test = X_test.reshape((X_test.shape[0], 1, self.window_size, self.no_of_features))
        MSE = self.model.evaluate(X_test, y_test)

        residuals = np.abs(np.round(y_test.T - prediction.T, 0))

        R2 = r2_score(y_test, prediction)

        y_predicted = np.round(prediction, 0)

        fitness = y_predicted.T == y_test.T
        good = np.sum(fitness)
        bad = fitness.shape[1] - np.sum(fitness)

        exact_accuracy = good / (good + bad)

        mean_error = np.mean(residuals)

        return MSE, R2, residuals, exact_accuracy, mean_error

    def plot_results(self, y_test, residuals, prediction):
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, residuals, color='red')
        plt.xlabel('predicted value')
        plt.ylabel('residual')
        plt.title('Results of CNN_LSTM Network - Residuals')

        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(prediction, y_test, color='blue')
        plt.xlabel('predicted value')
        plt.ylabel('real value')
        plt.title('Results of CNN_LSTM Network - Real vs Predicted')
        plt.show()
