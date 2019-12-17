import warnings
import numpy as np
from keras import regularizers, metrics
from keras.callbacks import callbacks
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
# from sklearn import metrics
import sklearn
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')


def ANN_classify(X_train, X_test, y_train, y_test, number_of_nodes, number_of_epochs, window_size, metrics,
                 regularization_penalty):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param number_of_nodes: number of neurons
    :param number_of_layers: number of layers
    :param dropout_rate: fraction rate between 0 and 1 of the input units to drop
    :param number_of_epochs: number of training iterations
    :param regularization_penalty: regularization penalty on layer parameters during optimization
    """

    # Split the generated dataset into train and test set
    no_of_features = X_train.shape[2]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    count_classes = y_train.shape[1]

    model = Sequential()
    model.add(LSTM(number_of_nodes, input_shape=(window_size, no_of_features),
                   kernel_regularizer=regularizers.l2(regularization_penalty)))

    # model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(count_classes, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)
    callback_list = [callbacks.EarlyStopping(monitor='val_loss', patience=20)]
    training_history = model.fit(X_train, y_train,
                                 batch_size=64, epochs=number_of_epochs,
                                 validation_data=(X_test, y_test), class_weight=class_weights, callbacks=callback_list)
    pred_train = model.predict_classes(X_train)
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores_train[1], 1 - scores_train[1]))
    pred_test = model.predict_classes(X_test)
    scores_test = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores_test[1], 1 - scores_test[1]))
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), pred_test)

    return scores_test, pred_test, confusion_matrix, model
