# Start Python Imports
import math
# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def Regression(X, y, sensor_name: str, score_name: str, degree, plot: str):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param sensor_name: name of the sensor user for plots
    :param score_name: name of the satisfaction score used for plots
    :param degree: degree of the polynomial
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    """

    # degree: determine the degrees of the polynomial function to be fitted and tested
    # degree = 1 will result in fitting a Linear Regression to the data

    # Split the generated dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    polynomial_transformation = PolynomialFeatures(degree=degree)

    # Fit polynomial regression
    X_polynomial = polynomial_transformation.fit_transform(X_train)
    polynomial_regression = LinearRegression().fit(X_polynomial, y_train)

    # Predict the satisfaction score
    y_predicted = polynomial_regression.predict(polynomial_transformation.fit_transform(X_test))

    X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, y_predicted)

    # Validate model
    MSE_regression = math.sqrt(mean_squared_error(y_test, y_predicted))
    R2_regression = r2_score(y_test, y_predicted)

    def plot_regression():
        # Plot original data
        plt.scatter(X_test, y_test, color='red')
        # Plot predicted regression function
        plt.plot(X_test_sorted, y_predicted_sorted, label=f"Degree {degree}," + f" $R^2$: {round(R2_regression, 3)}, "
                                                                                f"MSE: {round(MSE_regression, 3)}")
        plt.legend(loc='upper right')
        plt.xlabel(f"{sensor_name}")
        plt.ylabel(f"{score_name}")
        plt.title(f'Satisfaction vs indoor conditions (Polynomial Regression)')
        return

    if plot == 'plot':
        plot_regression()
    if plot == 'show':
        plot_regression()
        plt.show()
    return R2_regression, MSE_regression
