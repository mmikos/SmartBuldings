# Start Python Imports
import math
# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def Random_Forest_Regression(X, y, max_depth, n_estimators, plot, sensor_name: str, score_name: str):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param max_depth: maximum depth of the tree
    :param n_estimators: number of trees in the forest.
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    :param sensor_name: name of the sensor user for plots
    :param score_name: name of the satisfaction score used for plots
    """

    # Split the generated dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    random_forest_regression = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)

    random_forest_regression.fit(X_train, y_train)

    y_predicted = random_forest_regression.predict(X_test)

    X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, y_predicted)

    # Validate model
    R2_random_forest = r2_score(y_test, y_predicted)
    MSE_random_forest = math.sqrt(mean_squared_error(y_test, y_predicted))

    def plot_random_forest():
        # Plot original data
        plt.scatter(X_test, y_test, color='red')
        # Plot predicted regression function
        plt.plot(X_test_sorted, y_predicted_sorted, label=f"Depth {max_depth} with {n_estimators} trees, " +
                                                          f" $R^2$: {round(R2_random_forest, 3)},"
                                                          f" MSE: {round(MSE_random_forest, 3)}")
        plt.legend(loc='upper right')
        plt.xlabel(f"{sensor_name}")
        plt.ylabel(f"{score_name}")
        plt.title(f'Satisfaction vs indoor conditions (Random Forest Regression)')
        return

    if plot == 'plot':
        plot_random_forest()
    if plot == 'show':
        plot_random_forest()
        plt.show()
    return R2_random_forest, MSE_random_forest
