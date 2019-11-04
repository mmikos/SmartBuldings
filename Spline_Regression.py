# Start Python Imports
import math
# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Data Manipulation
import numpy as np
from patsy.highlevel import dmatrix
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def Spline_Regression(X, y, sensor_name: str, score_name: str, degree, quantiles, plot):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param sensor_name: name of the sensor user for plots
    :param score_name: name of the satisfaction score used for plots
    :param degree: degree of the polynomial
    :param quantiles: points of division of the series
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    """

    knots_array = np.quantile(X, quantiles)
    knots = tuple(knots_array)

    # Split the generated dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    # Generating cubic spline with 4 knots
    X_reshaped = dmatrix(f"bs(train, knots = {knots}, degree = {degree}, include_intercept = False)",
                         {"train": X_train}, return_type='dataframe')

    # Fitting Generalised linear model on transformed dataset
    spline_regression = LinearRegression().fit(X_reshaped, y_train)

    # Predict the satisfaction score
    y_predicted = spline_regression.predict(
        dmatrix(f"bs(test, knots = {knots},degree = {degree}, include_intercept = False)", {"test": X_test},
                return_type='dataframe'))

    X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, y_predicted)

    # Validate model
    R2_spline = r2_score(y_test, y_predicted)
    MSE_spline = math.sqrt(mean_squared_error(y_test, y_predicted))

    def plot_spline_regression():
        # Plot original data
        plt.scatter(X_test, y_test, color='red')
        # Plot predicted regression function
        plt.plot(X_test_sorted, y_predicted_sorted,
                 label=f"Degree {degree} with {len(quantiles)} knots, " + f"$R^2$: {round(R2_spline, 3)},"
                                                                          f" MSE: {round(MSE_spline, 3)}")
        plt.legend()
        plt.title('Satisfaction vs indoor conditions (Spline Regression)')
        plt.xlabel(f'{sensor_name}')
        plt.ylabel(f'{score_name}')
        # plt.show()
        return

    if plot == 'plot':
        plot_spline_regression()
    if plot == 'show':
        plot_spline_regression()
        plt.show()
    return R2_spline, MSE_spline
