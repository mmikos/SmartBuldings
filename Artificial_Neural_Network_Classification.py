# Start Python Imports
# Visualization
import matplotlib.pyplot as plt
import numpy as np
from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.layers import Dropout
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


def ANN_classify(X_train, X_test, y_train, y_test, number_of_nodes, number_of_layers, dropout_rate, number_of_epochs, regularization_penalty):
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
    layers_parameters = [number_of_nodes] * number_of_layers

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Shuffle the previously sorted dataset - in ANN ordered data impairs learning
    # X_standarized_shuffle, y_shuffle = shuffle(X_standarized, y)


    # Split the generated dataset into train and test set


    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.values[:, 0])

    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    count_classes = y_train.shape[1]
    print(count_classes)

    input_size = X_train.shape[1]

    # Create a model with 2 hidden layers and 32 neurons using Rectified Linear Unit Activation Function
    # Add regularization (L2 - norm) and dropout to fight overfitting

    def create_model(layers_parameters):
        model = Sequential()
        model.add(
            Dense(layers_parameters[0], activation='relu', kernel_regularizer=regularizers.l2(regularization_penalty),
                  input_dim=input_size))
        model.add(Dropout(dropout_rate))

        for neuron_number in layers_parameters[1:]:
            model.add(
                Dense(neuron_number, activation='relu', kernel_regularizer=regularizers.l2(regularization_penalty)))
            model.add(Dropout(dropout_rate))

        model.add(Dense(count_classes, activation='softmax', kernel_regularizer=regularizers.l2(regularization_penalty)))
        return model

    model = create_model(layers_parameters)
    

    # model.compile(optimizer='adam',
    #               loss='mean_squared_error',
    #               metrics=['mean_squared_error'])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    training_history = model.fit(X_train, y_train,
                                 batch_size=32, epochs=number_of_epochs,
                                 validation_data=(X_test, y_test), class_weight=class_weights)

    pred_train = model.predict(X_train)
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores_train[1], 1 - scores_train[1]))

    pred_test = model.predict(X_test)
    scores_test = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores_test[1], 1 - scores_test[1]))

    cofusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), pred_test.argmax(axis=1))

    # rows = cofusion_matrix.shape[0]
    # columns = cofusion_matrix.shape[1]
    matrix_percent = cofusion_matrix.astype(dtype=np.float32)

    # for column in range(columns):
    #     for row in range(rows):
    #         matrix_percent[row, column] = matrix_percent[row, column] / np.sum(cofusion_matrix, axis=1)[row]

    return scores_test, pred_test, matrix_percent
