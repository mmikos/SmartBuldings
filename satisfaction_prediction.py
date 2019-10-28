# Start Python Imports
import math

# Data Manipulation
import numpy as np
import pandas as pd
import statsmodels
from scipy.stats import pearsonr, jarque_bera, spearmanr, f_oneway, ttest_ind
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
import statsmodels.stats.power as smp
from satisfaction_score_data_generator import generate_dataset_with_sensor_readings_and_satisfaction_scores

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.layers import Dropout
from keras import regularizers

# Fitting a model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Neural Network
from keras.models import Sequential
from keras.layers import Dense

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
from patsy.highlevel import dmatrix
from tqdm import tqdm

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# %%

def sort_for_plotting(X, y):
    data_set = np.column_stack((X, y))
    test_set_sorted = data_set[data_set[:, 0].argsort()]
    X_sorted = test_set_sorted[:, 0]
    y_sorted = test_set_sorted[:, 1]

    return X_sorted, y_sorted


# %%

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
        plt.plot(X_test_sorted, y_predicted_sorted, label=f"Degree {deg}," + f" $R^2$: {round(R2_regression, 3)}, "
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
    return R2_regression, MSE_regression,


# %%

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


# %%

def Random_Forest_Regression(X, y, max_depth, n_estimators, plot):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param max_depth: maximum depth of the tree
    :param n_estimators: number of trees in the forest.
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
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


# %%

def Decision_Tree_Regression(X, y, max_depth, n_estimators, plot):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param max_depth: maximum depth of the tree
    :param n_estimators: maximum number of estimators at which boosting is terminated
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
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


# %%

def ANN(X, y, number_of_nodes, number_of_layers, dropout_rate, number_of_epochs, regularization_penalty, plot):
    """
    :param X: array of independent variables
    :param y: array of dependent variable
    :param number_of_nodes: number of neurons
    :param number_of_layers: number of layers
    :param dropout_rate: fraction rate between 0 and 1 of the input units to drop
    :param number_of_epochs: number of training iterations
    :param regularization_penalty: regularization penalty on layer parameters during optimization
    :param plot: takes values 'plot' to plot the functions and 'show' to plot and display
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

        # plt.plot(training_history.history['mean_squared_error'])
        # plt.plot(training_history.history['val_mean_squared_error'])
        # plt.title('Model Error')
        # plt.ylabel('MSE')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'])
        # plt.show()

    if plot == 'plot':
        plot_artificial_neural_network()
    if plot == 'show':
        plot_artificial_neural_network()
        plt.show()

    return R2_ANN, MSE_ANN, training_history


# %%
# Generate dataset with the scores and reading using satisfaction_score_data_generator.py
# or read CSV
number_of_samples = 100
noise_standard_deviation = 0
wavelet = 'db8'

satisfaction_vs_sensors = generate_dataset_with_sensor_readings_and_satisfaction_scores(number_of_samples,
                                                                                        noise_standard_deviation,
                                                                                        wavelet)
# satisfaction_vs_sensors = pd.read_csv('satisfaction_vs_sensors.csv')
# satisfaction_vs_sensors.to_csv('satisfaction_vs_sensors_smoothing.csv', sep=',')
# %%
# Display summary statistics
summary_statistics = satisfaction_vs_sensors.describe()

# Plot data distribution
no_of_columns = np.shape(satisfaction_vs_sensors)[1]
measurements = satisfaction_vs_sensors.columns.values

plt.figure(figsize=(4.5*no_of_columns,5*no_of_columns))
for i in range(0,len(measurements)):
    plt.subplot(no_of_columns + 1, no_of_columns, i + 1)
    sns.distplot(satisfaction_vs_sensors[measurements[i]],kde=True)

# Display Box Plot
plt.figure(figsize=(no_of_columns,5*no_of_columns))
for i in range(0,len(measurements)):
    plt.subplot(no_of_columns + 1, no_of_columns, i + 1)
    sns.boxplot(satisfaction_vs_sensors[measurements[i]],color='lightblue',orient='v')
    plt.tight_layout()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(satisfaction_vs_sensors.corr(), cmap='Blues', annot=True)
plt.show()
# %%
# Specify the name of the measurement/score of interest
# Choose from: temperature, humidity, heat_index, air, light, noise

measurement_name = 'air'
# measurement_name2 = 'humidity'
# measurement_name3 = 'temperature'
sensor_name = f'sensor_{measurement_name}'
# sensor_name2 = f'sensor_{measurement_name2}'
# sensor_name3 = f'sensor_{measurement_name3}'
score_name = f'comfort_score_{measurement_name}'

satisfaction_vs_sensors_sorted = satisfaction_vs_sensors.sort_values([sensor_name])

# Divide the dataset into predictors and outcome
X = satisfaction_vs_sensors[[sensor_name]]
# X = satisfaction_vs_sensors[[sensor_name, sensor_name2, sensor_name3]]
y = satisfaction_vs_sensors[[score_name]]

# Plot the chosen variables
X_sorted, y_sorted = sort_for_plotting(X, y)
plt.figure(figsize=(12, 10))
plt.plot(X_sorted, y_sorted)
plt.xlabel(f"{sensor_name}")
plt.ylabel(f"{score_name}")
plt.title(f'Satisfaction vs indoor conditions (Generated data)')
plt.show()
# %%
# H0: Data is normally distributed
# H1: Skewness and kurtosis doesn't match a normal distribution

statistics_jb_X, p_value_jb_X = jarque_bera(X)
statistics_jb_y, p_value_jb_y = jarque_bera(y)

# Plot variable to examine its distribution
sns.distplot(y)

# If the variable doesn't have a normal distribution check if it has log-normal distribution
X_log = np.log(X)
y_log = np.log(X_log)

statistics_jb_X_log, p_value_jb_X_log = jarque_bera(X_log)
statistics_jb_y_log, p_value_jb_y_log = jarque_bera(y_log)

# Plot a transformed variable to examine its distribution
sns.distplot(y_log)

print(f'Jarque-Bera Test for X: statistics = {statistics_jb_X}; p-value = {p_value_jb_X}')
print(f'Jarque-Bera Test for y: statistics = {statistics_jb_y}; p-value = {p_value_jb_y}')

print(f'Jarque-Bera Test for X_log: statistics = {statistics_jb_X_log}; p-value = {p_value_jb_X_log}')
print(f'Jarque-Bera Test for y_log: statistics = {statistics_jb_y_log}; p-value = {p_value_jb_y_log}')
# %%
# H0: There is no correlation
# H1: Correlation coefficient is statistically significant

rho, p_value_spearman = spearmanr(X, y)
r, p_value_pearson = pearsonr(X.iloc[:, 0], y.iloc[:, 0])

print(f'Correlation Spearman = {rho}, p-value = {p_value_spearman}')
print(f'Correlation Pearson = {r}, p-value = {p_value_pearson}')
# %%
# Set parameters to perform the sample size estimation using L-value
R2_regression = 0.4
number_of_predictors_degrees = 3

# Read from the tables e.g. for 80% power and 3 predictors the value is:
L_value = 10.9

# Effect size can be calculated using the R2 statistic
cohen_effect_size = R2_regression / (1 - R2_regression)

estimated_sample_size = (L_value / cohen_effect_size) + number_of_predictors_degrees + 1
# %%
# Set parameters to perform the power analysis t-test, put "None" by the parameter that is needed to be calculated

R2_regression = 0.4
effect_size = R2_regression / (1 - R2_regression)
number_of_observations = None
significance = 0.05
test_power = 0.9

power_test = smp.TTestPower().solve_power(effect_size=effect_size, nobs=number_of_observations, alpha=significance,
                                    power = test_power, alternative='larger')

print(f'Sample Size: {power_test}')
# %%
# Plots how the power changes with number of observations and effect size

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(2,1,1)
fig = smp.TTestPower().plot_power(dep_var='nobs',
                                  nobs= np.arange(2, 120),
                                  effect_size=np.array([0.4, 0.6, 0.8, 1, 1.5, 2]),
                                  alternative='larger',
                                  ax=ax, title='T-Test Power')
ax = fig.add_subplot(2,1,2)
fig = smp.TTestPower().plot_power(dep_var='es',
                                  nobs=np.array([10, 20, 40, 60, 80, 100]),
                                  effect_size=np.linspace(0.1, 2),
                                  alternative='larger',
                                  ax=ax, title='')
# %%
# F-statistic:
#     H0: All of the regression coefficients are equal to zero
#     H1: Model has good predictive capability
# T-statistics:
#     H0: Variable is not signoificant to the model
#     H1: Significant relationship between the response and explanatory variables

degree = 3

# Perform polynomial transformation
polynomial_features = PolynomialFeatures(degree=degree)
X_polynomial = polynomial_features.fit_transform(X_log)

# Fit the model
model = sm.OLS(y, X_polynomial).fit()
y_predicted = model.predict(X_polynomial)

slope, intercept = np.polyfit(np.log(X.iloc[:, 0]), np.log(y.iloc[:, 0]), 1)

# Plot the prediction
plt.scatter(X,y)
X_sorted, y_sorted_p = sort_for_plotting(X, y_predicted)
plt.plot(X_sorted, y_sorted_p)
plt.show()

# Show summary statistics
model.summary()
# %%
# H0: Homoskedasticity
# H1: Heteroskedasticity

bp_test = het_breuschpagan(model.resid, X_polynomial)
white_test = het_white(model.resid,  model.model.exog)

labels = ['Lagrange Multiplier Statistic', 'Lagrange Multiplier-Test p-value', 'F-Statistic', 'F-Test p-value']

print(dict(zip(labels, bp_test)))
print(dict(zip(labels, white_test)))

# %%
t_statistics, p_value_t = ttest_ind(X, y)

# The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
# 1, The samples are independent.
# 2. Each sample is from a normally distributed population.
# 3. The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
# H0:  Two or more groups have the same population mean
F_statistic, p_value_anova = f_oneway(X, y)
print(f'ANOVA p-value = {p_value_anova}')

# %%
# Random Forest
# Chose the maximum depth of the tree and number of trees in the forest.
max_depth = [4]
n_estimators = [50]
Results_Random_Forest_list = []

plt.figure(figsize=(12, 10))
for depth in tqdm(max_depth):
    for estimator in n_estimators:
        R2_random_forest, MSE_random_forest = Random_Forest_Regression(X, y, depth, estimator, 'plot')
        Results_Random_Forest_list.append([depth, estimator, R2_random_forest, MSE_random_forest])

plt.show()
Results_Random_Forest = pd.DataFrame(Results_Random_Forest_list, columns=['max_depth', 'n_estimators',
                                                                          'R2_random_forest', 'MSE_random_forest'])
Results_Random_Forest.sort_values(['R2_random_forest'], ascending=False)
# %%
# Decision Tree
# Chose the maximum depth of the tree and number of trees in the forest.

max_depth = [4]
n_estimators = [50]
Results_Decision_Tree_list = []

plt.figure(figsize=(12, 10))
for depth in tqdm(max_depth):
    for estimator in n_estimators:
        R2_decision_tree, MSE_decision_tree = Decision_Tree_Regression(X, y, depth, estimator, 'plot')
        Results_Decision_Tree_list.append([depth, estimator, R2_decision_tree, MSE_decision_tree])

plt.show()
Results_Decision_Tree = pd.DataFrame(Results_Decision_Tree_list, columns=['max_depth', 'n_estimators',
                                                                          'R2_decision_tree', 'MSE_decision_tree'])
Results_Decision_Tree.sort_values(['R2_decision_tree'], ascending=False)
# %%
# ANN
# Chose the number of neurons, number of hidden layers, a dropout rate and regularization penalty

number_of_nodes = [100]
number_of_layers = [2]
dropout_rate = [0.2]
number_of_epochs = [1000]
regularization_penalty = [0.0001]
Results_ANN_list = []

# Fit ANN model
plt.figure(figsize=(12, 10))

for nodes in tqdm(number_of_nodes):
    for layers in number_of_layers:
        for dropout in dropout_rate:
            for epochs in number_of_epochs:
                for regularization in regularization_penalty:
                    R2_ANN, MSE_ANN, training_history = ANN(X, y, nodes, layers, dropout, epochs, regularization,
                                                            'show')
                    Results_ANN_list.append([nodes, layers, dropout, epochs, regularization, R2_ANN, MSE_ANN])

# plt.show()
Results_ANN = pd.DataFrame(Results_ANN_list, columns=['Number of nodes', 'Number of layers', 'Dropout rate',
                                                      'Number of epochs', 'Regularization penalty', 'R2_ANN',
                                                      'MSE_ANN'])
Results_ANN.sort_values(['R2_ANN'], ascending=False)
# %%

# Results_ANN.to_csv('Results_ANN.csv', sep=',')
# %%
# Evaluate performance of models over different degrees
# Regression

degree_range_regression = 8
Results_regression_list = []


plt.figure(figsize=(12, 10))
for deg in range(2, degree_range_regression):
    R2_regression, MSE_regression = Regression(X, y, sensor_name, score_name, deg,
                                               'plot')
    Results_regression_list.append([deg, R2_regression, MSE_regression])

plt.show()
Results_regression = pd.DataFrame(Results_regression_list, columns=['deg', 'R2_regression', 'MSE_regression'])
Results_regression.sort_values(['R2_regression'], ascending=False)
# %%
# Spline Regression
degree_range_spline = 15
quantiles_in_spline_regression = (0.25, 0.5, 0.75)
Results_splines_list = []

plt.figure(figsize=(12, 10))
for deg in range(2, degree_range_spline):
    R2_spline, MSE_spline = Spline_Regression(X, y, sensor_name, score_name, deg,
                                              quantiles_in_spline_regression, 'plot')
    Results_splines_list.append([deg, R2_spline, MSE_spline])

plt.show()
Results_splines = pd.DataFrame(Results_splines_list, columns=['deg', 'R2_spline', 'MSE_spline'])
Results_splines.sort_values(['R2_spline'], ascending=False)