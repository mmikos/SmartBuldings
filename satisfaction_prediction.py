# Visualization
import matplotlib.pyplot as plt
from Sort import sort_for_plotting
# Data Manipulation
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.power as smp
from scipy.stats import pearsonr, jarque_bera, spearmanr, f_oneway, ttest_ind
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
# Fitting a model
from Artificial_Neural_Network import ANN
from Decision_Tree_Regression import Decision_Tree_Regression
from Random_Forest_Regression import Random_Forest_Regression
from Regression import Regression
from Spline_Regression import Spline_Regression
from satisfaction_score_data_generator import generate_dataset_with_sensor_readings_and_satisfaction_scores

from tqdm import tqdm
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

# %%

# Generate dataset with the scores and reading using satisfaction_score_data_generator.py
# or read CSV
number_of_samples = 100
signal_to_noise_ratio = 1
wavelet = 'db8'

satisfaction_vs_sensors, satisfaction_vs_sensors_null = generate_dataset_with_sensor_readings_and_satisfaction_scores(
    number_of_samples, signal_to_noise_ratio, wavelet)
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
        R2_random_forest, MSE_random_forest = Random_Forest_Regression(X, y, depth, estimator, 'plot', sensor_name,
                                                                       score_name)
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
        R2_decision_tree, MSE_decision_tree = Decision_Tree_Regression(X, y, depth, estimator, 'plot', sensor_name,
                                                                       score_name)
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
                                                            'show', sensor_name, score_name)
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
for degree in range(2, degree_range_regression):
    R2_regression, MSE_regression = Regression(X, y, sensor_name, score_name, degree,
                                               'plot')
    Results_regression_list.append([degree, R2_regression, MSE_regression])

plt.show()
Results_regression = pd.DataFrame(Results_regression_list, columns=['degree', 'R2_regression', 'MSE_regression'])
Results_regression.sort_values(['R2_regression'], ascending=False)
# %%
# Spline Regression
degree_range_spline = 15
quantiles_in_spline_regression = (0.25, 0.5, 0.75)
Results_splines_list = []

plt.figure(figsize=(12, 10))
for degree in range(2, degree_range_spline):
    R2_spline, MSE_spline = Spline_Regression(X, y, sensor_name, score_name, degree,
                                              quantiles_in_spline_regression, 'plot')
    Results_splines_list.append([degree, R2_spline, MSE_spline])

plt.show()
Results_splines = pd.DataFrame(Results_splines_list, columns=['degree', 'R2_spline', 'MSE_spline'])
Results_splines.sort_values(['R2_spline'], ascending=False)