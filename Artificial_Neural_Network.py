# Start Python Imports
# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Preprocessing
from sklearn.utils import shuffle
from keras.layers import Dropout
from keras import regularizers
# Neural Network
from keras.models import Sequential
from keras.layers import Dense
# Preprocessing
from sklearn import preprocessing
# Evaluation
from sklearn.metrics import r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def ANN(X, y, number_of_nodes, number_of_layers, dropout_rate, number_of_epochs, regularization_penalty, plot,
        sensor_name: str, score_name: str):
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
    X_standarized = scaler.fit_transform(X)

    # Shuffle the previously sorted dataset - in ANN ordered data impairs learning
    X_standarized_shuffle, y_shuffle = shuffle(X_standarized, y)

    # Split the generated dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_standarized_shuffle, y_shuffle, test_size=1 / 3,
                                                        random_state=42, shuffle=True)
    input_size = X.shape[1]

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

        model.add(Dense(input_size, activation='relu', kernel_regularizer=regularizers.l2(regularization_penalty)))
        return model

    model = create_model(layers_parameters)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    training_history = model.fit(X_train, y_train,
                                 batch_size=32, epochs=number_of_epochs,
                                 validation_data=(X_test, y_test))

    y_predicted = model.predict(X_test, batch_size=32, verbose=0)

    X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, y_predicted)

    MSE_ANN = model.evaluate(X_test, y_test)[1]

    R2_ANN = r2_score(y_test, y_predicted)

    # predicted_set = np.c_[X_standarized, y_predicted]

    def plot_artificial_neural_network():
        plt.figure(figsize=(12, 10))
        # Plot original data
        plt.scatter(scaler.inverse_transform(X_test), y_test, color='red')
        # Plot predicted regression function
        plt.plot(scaler.inverse_transform(X_test_sorted), y_predicted_sorted,
                 label=f"Dropout {dropout_rate} with {number_of_epochs} epochs and {regularization_penalty} penalty, " +
                       f"Nodes: {number_of_nodes}, Layers: {number_of_layers}, " +
                       f"$R^2$: {round(R2_ANN, 3)}, MSE: {round(MSE_ANN, 3)}")
        plt.legend()
        plt.title('Satisfaction vs indoor conditions (ANN)')
        plt.xlabel(f'{sensor_name}')
        plt.ylabel(f'{score_name}')

    if plot == 'plot':
        plot_artificial_neural_network()
    if plot == 'show':
        plot_artificial_neural_network()
        plt.show()

    return R2_ANN, MSE_ANN, training_history
