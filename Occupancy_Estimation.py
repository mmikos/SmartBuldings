# Data manipulation
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# Visualization
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Models
from sklearn.model_selection import train_test_split
import matplotlib.pylab as pylab

import Regression
from Sort import sort_for_plotting

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'small'}
pylab.rcParams.update(params)

import warnings

warnings.filterwarnings('ignore')

# read the dataset
measure = pd.read_csv('data_sets/OLY-A-417.csv', sep=';', decimal='.')

room_5 = pd.read_csv('data_sets/Room_5.csv', sep=';', decimal='.')

co2 = measure[['timestamp(Europe/Berlin)', 'co2']]

date_co2 = co2[(co2['timestamp(Europe/Berlin)'] >= '24/10/2019 07:40') & (co2['timestamp(Europe/Berlin)'] <= '25/10/2019')]

no_occupants = room_5[['date_time', 'no_occupants']]
# # measure = pd.read_csv('data_sets/OLY-A-415.csv', sep=',', decimal='.')
date_no_occupants = no_occupants[(no_occupants['date_time'] >= '24/10/2019') & (no_occupants['date_time'] <= '25/09/2019')]

date_no_occupants = date_no_occupants.sort_values(by = ['date_time'])
# date_co2 = date_co2.sort_values(by = ['timestamp(Europe/Berlin)'])

# plt.figure(figsize=(12, 10))
# plt.plot(date_no_occupants.iloc[:, 0], date_no_occupants.iloc[:, 1], label = 'Occupancy')
# plt.legend()
# plt.show()

# fig, axs = plt.subplots(2)
# fig.suptitle('Aligning x-axis using sharex')
# axs[0].plot(date_no_occupants.iloc[:, 0], date_no_occupants.iloc[:, 1])
# axs[1].plot(date_co2.iloc[:, 0], date_co2.iloc[:, 1])
# plt.show()

data_bricks = pd.read_csv('data_sets/export_room3.csv', sep=';', decimal='.')
data_bricks = data_bricks.sort_values(by = ['Datetime'])
# data_bricks = data_bricks.groupby(['Datetime']).mean()
data_bricks.groupby(['Datetime', 'SpaceName'])['value'].mean().reset_index()

data_bricks = data_bricks[(data_bricks['Datetime'] >= '24/10/2019') & (data_bricks['Datetime'] <= '25/10/2019 ')]

# data_bricks2 = data_bricks.loc[data_bricks['SpaceName'] == 'Room 4.3', ['Datetime', 'SpaceName', 'Value']]

plt.figure(figsize=(12, 10))
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(pd.to_datetime(data_bricks.iloc[:, 0], format='%d/%m/%Y %H:%M').dt.time , data_bricks.iloc[:, 4], color=color)
ax1.plot(pd.to_datetime(date_no_occupants.iloc[:, 0], format='%d/%m/%Y %H:%M').dt.time, date_no_occupants.iloc[:, 1], color='green')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CO2', color=color)

ax2.plot(pd.to_datetime(date_co2.iloc[:, 0], format='%d/%m/%Y %H:%M').dt.time, date_co2.iloc[:, 1].shift(-1), color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title(f'Occupancy')
plt.legend()
plt.show()

reg_data = pd.read_csv('data_sets/OLY-A-417.csv', sep=';', decimal='.')
reg_data = reg_data[(reg_data['timestamp(Europe/Berlin)'] >= '24/10/2019') &
                    (reg_data['timestamp(Europe/Berlin)'] <= '25/10/2019 ')]

from sklearn import linear_model
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, PolynomialFeatures
from sklearn.cluster import DBSCAN, KMeans

# outlier_detection = DBSCAN(eps = .2, metric = 'euclidean', min_samples = 5, n_jobs = -1)
outlier_detection = KMeans(n_clusters=7, random_state=0)

reg_data_no_na = reg_data.dropna()

import seaborn as sns
sns.boxplot(x=reg_data_no_na[['co2']])
plt.show()

# z = np.abs(stats.zscore(reg_data_no_na[['co2']]))
# print(z)
# threshold = 3
# print(np.where(z > 3))

# reg_data_no_na = reg_data_no_na[(z < 3).all(axis=1)]

scaler_sound = StandardScaler()
scaler_co2 = StandardScaler()

sound = scaler_sound.fit_transform(reg_data_no_na[['noise']])
# sound = reg_data_no_na[['timestamp(Europe/Berlin)']]
co2 = scaler_co2.fit_transform(reg_data_no_na[['co2']])

outlier_set = np.column_stack((sound, co2))

clusters = outlier_detection.fit_predict(co2)

from matplotlib import cm
# cmap = cm.get_cmap('Set1')
outlier_set[:, 0] = scaler_sound.inverse_transform(outlier_set[:, 0])
outlier_set[:, 1] = scaler_co2.inverse_transform(outlier_set[:, 1])
plt.scatter(outlier_set[:, 0], outlier_set[:, 1], c=clusters, cmap='viridis')
plt.legend()
plt.show()

# reg_data_no_na = reg_data_no_na - np.mean(reg_data_no_na)

scaler_predictors = StandardScaler()
scaler_output = StandardScaler()
# reg_data2 = preprocessing.normalize(reg_data_no_na, norm='l2')
X = scaler_predictors.fit_transform(reg_data_no_na[['temp', 'humid', 'light', 'noise']])
y = scaler_output.fit_transform(reg_data_no_na[['co2']])
# X = reg_data_no_na[['temp', 'humid', 'light', 'noise']]
# y = reg_data_no_na[['co2']]

import statsmodels.api as sm
degree = 1
# sensor_values_centered = sensor_values - np.mean(sensor_values)

# Perform polynomial transformation
polynomial_features = PolynomialFeatures(degree=degree)
sensor_values_polynomial = polynomial_features.fit_transform(X)

# Fit the model
model = sm.OLS(y, sensor_values_polynomial).fit()
# fit_regularized(alpha=0.2, L1_wt=0.5)

score_values_predicted = model.predict(sensor_values_polynomial)

print(model.summary())

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X, y)

params = np.append(clf.intercept_, clf.coef_)
predictions = clf.predict(X)

newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (np.sum((y.T-predictions.T)**2))/(len(newX)-len(newX.columns))

var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/sd_b

p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

sd_b = np.round(sd_b, 3)
ts_b = np.round(ts_b, 3)
p_values = np.round(p_values, 3)
params = np.round(params, 4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
                                                                                             p_values]
print(myDF3)

rsquared = clf.score(X, y)
F = rsquared / ((1 - rsquared) / (len(y) - 1 - 1))
p_value = stats.f.sf(F, 1, len(y) - 1 - 1)

X = scaler_predictors.inverse_transform(X)
y = scaler_output.inverse_transform(y)
predictions = scaler_output.inverse_transform(predictions.reshape(-1, 1))

X_test_sorted, y_predicted_sorted = sort_for_plotting(X[:, 0], predictions)
# Plot original data
plt.scatter(X[:, 0], y, color='red')
# plt.scatter(X[:, 1], y, color='blue')
# plt.scatter(X[:, 2], y, color='green')
# plt.scatter(X[:, 3], y, color='yellow')
# Plot predicted regression function

plt.plot(X_test_sorted, y_predicted_sorted, label=f"Degree {1}," + f" $R^2$: {round(rsquared, 3)}, MSE: {round(MSE, 3)}")

plt.legend(loc='upper right')
plt.xlabel("Temp")
plt.ylabel("CO2")
plt.title(f'Satisfaction vs indoor conditions (Polynomial Regression)')
plt.show()



# def get_single_measurement(data, measurement=str):
#     measurement_name = data.loc[data['PortName'] == f'{measurement}', ['Datetime', 'PortName', 'Value']].pivot_table(
#         index='Datetime', columns='PortName', values='Value')
#
#     return measurement_name


# Temperature = get_single_measurement(data_bricks, 'Temperature')
# Humidity = get_single_measurement(data_bricks, 'Humidity')
# Light = get_single_measurement(data_bricks, 'Light')
# Noise = get_single_measurement(data_bricks, 'Sound')
# Occupancy = get_single_measurement(data_bricks, 'Occupancy')
# Movement = get_single_measurement(data_bricks, 'Motion')
# CO2 = get_single_measurement(data_bricks, 'CarbonDioxide')

# measurement_name = data_bricks.loc[data_bricks['PortName'] == 'Occupancy', ['Datetime', 'PortName', 'SpaceName',
#                                                                             'Value']].pivot_table(index='Datetime',
#                                                                                                   columns='SpaceName',
#                                                                                                   values='Value')

print('whatever')
