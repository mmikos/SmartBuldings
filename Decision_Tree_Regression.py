# Start Python Imports
import math
# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def Decision_Tree_Regression(X, y, max_depth, n_estimators, plot, sensor_name: str, score_name: str):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param max_depth: maximum depth of the tree
    :param n_estimators: maximum number of estimators at which boosting is terminated
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
    :param sensor_name: name of the sensor user for plots
    :param score_name: name of the satisfaction score used for plots
    """

    # Split the generated dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

    boost_decision_tree_regression = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                                       n_estimators=n_estimators)

    boost_decision_tree_regression.fit(X_train, y_train)

    y_predicted = boost_decision_tree_regression.predict(X_test)

    X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, y_predicted)

    # Validate model
    R2_decision_tree = r2_score(y_test, y_predicted)
    MSE_decision_tree = math.sqrt(mean_squared_error(y_test, y_predicted))

    def plot_decision_tree():
        # Plot original data
        plt.scatter(X_test, y_test, color='red')
        # Plot predicted regression function
        plt.plot(X_test_sorted, y_predicted_sorted, label=f"Depth {max_depth} with {n_estimators} estimators" +
                                                          f" $R^2$: {round(R2_decision_tree, 3)}, "
                                                          f"MSE: {round(MSE_decision_tree, 3)}")
        plt.legend(loc='upper right')
        plt.xlabel(f"{sensor_name}")
        plt.ylabel(f"{score_name}")
        plt.title(f'Satisfaction vs indoor conditions (Boosted Decision Tree Regression)')
        return

    if plot == 'plot':
        plot_decision_tree()
    if plot == 'show':
        plot_decision_tree()
        plt.show()
    return R2_decision_tree, MSE_decision_tree
