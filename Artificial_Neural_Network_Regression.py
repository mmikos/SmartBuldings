# Start Python Imports
# Visualization
import matplotlib.pyplot as plt
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


def ANN_regress(data, number_of_nodes, dropout_rate, window_size, batch_size, number_of_epochs, regularization_penalty):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param number_of_nodes: number of neurons
    :param number_of_layers: number of layers
    :param dropout_rate: fraction rate between 0 and 1 of the input units to drop
    :param number_of_epochs: number of training iterations
    :param regularization_penalty: regularization penalty on layer parameters during optimization
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    :param sensor_name: name of the sensor user for plots
    :param score_name: name of the satisfaction score used for plots
    """

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = data.to_numpy()
    data[:, 0:2] = scaler.fit_transform(data[:, 0:2])

    def prepare_data(data, window_size):
        batches = []
        labels = []
        for idx in range(len(data) - window_size - 1):
            batches.append(data[idx: idx + window_size, 0:2])
            labels.append(data[idx + window_size, 2])
        return np.array(batches), np.array(labels)

    batches, labels = prepare_data(data, window_size)

    X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42, shuffle=True)

    # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    # X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create a model with 2 hidden layers and 32 neurons using Rectified Linear Unit Activation Function
    # Add regularization (L2 - norm) and dropout to fight overfitting

    model = Sequential()

    model.add(layers.LSTM(number_of_nodes, input_shape=(window_size, batches.shape[2])))

    model.add(layers.Dense(1))

    opt = SGD(learning_rate=0.001, momentum=0.99)

    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    training_history = model.fit(X_train, y_train,
                                 batch_size=batch_size, epochs=number_of_epochs, validation_data=(X_test, y_test))

    y_predicted = model.predict(X_test, batch_size=batch_size, verbose=2)

    y_predicted = np.round(y_predicted, 0)
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    MSE_ANN = model.evaluate(X_test, y_test)[1]

    residuals = y_test.T - y_predicted.T

    R2_ANN = r2_score(y_test, y_predicted)
    plt.figure(figsize=(12, 10))
    # Plot original data
    plt.scatter(y_predicted, residuals, color='red')
    plt.xlabel('predicted value')
    plt.ylabel('residual')
    plt.show()

    return R2_ANN, MSE_ANN, y_predicted, training_history
