# Start Python Imports
# Visualization
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import callbacks

from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, LSTM
from keras import regularizers
# Neural Network
from keras.models import Sequential
from keras.layers import Dense
# Preprocessing
from sklearn import preprocessing, metrics
# Evaluation
from sklearn.metrics import r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def ANN_classify(X_train, X_test, y_train, y_test, number_of_nodes, number_of_epochs, window_size):
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

    # Split the generated dataset into train and test set
    no_of_features = 2
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    count_classes = y_train.shape[1]
    print(count_classes)

    input_size = X_train.shape[1]

    model = Sequential()
    model.add(LSTM(number_of_nodes, activation='relu', input_shape=(window_size, no_of_features)))

    # model.add(Dropout(dropout_rate))
    # model.add(Flatten())
    model.add(Dense(count_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    training_history = model.fit(X_train, y_train,
                                 batch_size=64, epochs=number_of_epochs,
                                 validation_data=(X_test, y_test), class_weight=class_weights, callbacks=callback_list)
    pred_train = model.predict_classes(X_train)
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores_train[1], 1 - scores_train[1]))
    pred_test = model.predict_classes(X_test)
    scores_test = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores_test[1], 1 - scores_test[1]))
    confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), pred_test)

    return scores_test, pred_test, confusion_matrix
