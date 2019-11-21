# Data manipulation
import json
import datetime

import matplotlib.pylab as pylab
# Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.io.json import json_normalize
# Preprocessing
from pandas.io.json import json_normalize
from sklearn.metrics import r2_score
# Models
from sklearn.model_selection import train_test_split

from Sort import sort_for_plotting

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'small'}
pylab.rcParams.update(params)

import warnings
import urllib
warnings.filterwarnings('ignore')

# "devices":
# "name": "OLY-A-414",
# "uuid": "awair-omni_9049",
# "timezone": "Europe/Amsterdam",
# "deviceType": "awair-omni",
# "deviceId": 9049
#
# "name": "OLY-A-416",
# "uuid": "awair-omni_7675",
# "timezone": "Europe/Amsterdam",
# "deviceType": "awair-omni",
# "deviceId": 7675
#
# "name": "OLY-A-413",
# "uuid": "awair-omni_9033",
# "timezone": "Europe/Amsterdam",
# "deviceType": "awair-omni",
# "deviceId": 9033
#
# "name": "OLY-A-415",
# "uuid": "awair-omni_8989",
# "timezone": "Europe/Amsterdam",
# "deviceType": "awair-omni",
# "deviceId": 8989
#
# "name": "OLY-A-417",
# "uuid": "awair-omni_7663",
# "timezone": "Europe/Amsterdam",
# "deviceType": "awair-omni",
# "deviceId": 7663

start_date = "2019-11-19 09:00"
end_date = "2019-11-20 17:30"

start_url = urllib.parse.quote(start_date)
end_url = urllib.parse.quote(end_date)

response = requests.get("https://edgetech.avuity.com/VuSpace/api/report-occupancy-by-area/index?access-token"
                        f"=Futo24i1PcUZ_HnZ&startTs={start_url}&endTs={end_url}")

# print(response.status_code)

occupancy_json = response.json()
occupancy_str = json.dumps(occupancy_json)
occupancy_data_dict = json.loads(occupancy_str)
occupancy_data_normalised = json_normalize(occupancy_data_dict['items'])
occupancy = pd.DataFrame.from_dict(occupancy_data_normalised)

# AWAIR

start_iso = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M').isoformat()
end_iso = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M').isoformat()

headers = {
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNTE4MDgifQ.HnZ_258AsEfbYLzmpK_g4jbTItIYbEQh_UaxCDO0S88',
}

params = (
    ('from', start_iso),
    ('to', end_iso),
)

response_awair = requests.get('http://developer-apis.awair.is/v1/orgs/1097/devices/awair-omni/7663/air-data/15-min'
                              '-avg', headers=headers, params=params)

# print(response_awair.status_code)

awair_json = response_awair.json()
awair_str = json.dumps(awair_json)
awair_data_dict = json.loads(awair_str)
awair = pd.io.json.json_normalize(awair_data_dict["data"], record_path="sensors", meta=['timestamp', 'score'])

room = 'ROOM 4'

occupancy_selected = occupancy.loc[occupancy['areaName'] == f'{room}', ['startTs', 'occupancy']]

# date_occupancy_selected = occupancy_selected[(occupancy_selected['startTs'] >= '2019-11-19 09:00')
#                                              & (occupancy_selected['startTs'] <= '2019-11-20 17:35')]
# date_occupancy_selected.iloc[:, 0] = pd.to_datetime(date_occupancy_selected.iloc[:, 0], format='%Y-%m-%d %H:%M')

date_occupancy_selected = occupancy_selected.set_index('startTs')

date_occupancy_selected.index = pd.to_datetime(date_occupancy_selected.index)

date_occupancy_agg = date_occupancy_selected.resample('15T').max().ffill().astype(int)

# read the dataset
# measurements = pd.read_csv('data_sets/19_20/OLY-A-415.csv', sep=',', decimal='.')
# co2 = measurements[['timestamp(Europe/Berlin)', 'co2', 'noise']]
# date_co2 = co2[(co2['timestamp(Europe/Berlin)'] >= '2019-11-19 09:00')
#                & (co2['timestamp(Europe/Berlin)'] <= '2019-11-21 17:45')]

awair.iloc[:, 2] = pd.to_datetime(awair.iloc[:, 2], format='%Y-%m-%d %H:%M')

awair.sort_values(by = 'timestamp', ascending = True)

co2 = awair.loc[awair['comp'] == 'co2', ['timestamp', 'value']]
noise = awair.loc[awair['comp'] == 'spl_a', ['timestamp', 'value']]


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(date_occupancy_agg.reset_index().iloc[:, 0],
         date_occupancy_agg.reset_index().iloc[:, 1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CO2', color=color)

ax2.plot(co2.iloc[:, 0], co2.iloc[:, 1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs CO_2')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(date_occupancy_agg.reset_index().iloc[:, 0],
         date_occupancy_agg.reset_index().iloc[:, 1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('noise', color=color)

ax2.plot(noise.iloc[:, 0], noise.iloc[:, 1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs Noise')
plt.legend()
plt.show()

reg_data = pd.read_csv('data_sets/OLY-A-417.csv', sep=';', decimal='.')
reg_data = reg_data[(reg_data['timestamp(Europe/Berlin)'] >= '24/10/2019') &
                    (reg_data['timestamp(Europe/Berlin)'] <= '25/10/2019 ')]

from sklearn import linear_model
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans

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
X = reg_data_no_na[['temp', 'noise']]
y = reg_data_no_na[['co2']]
# X = reg_data_no_na[['temp', 'humid', 'light', 'noise']]
# y = reg_data_no_na[['co2']]

import statsmodels.api as sm
degree = 1
# sensor_values_centered = sensor_values - np.mean(sensor_values)

# Perform polynomial transformation
polynomial_features = PolynomialFeatures(degree=degree)
sensor_values_polynomial = polynomial_features.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(sensor_values_polynomial, y, test_size=1 / 3, random_state=42)

# Fit the model
model = sm.OLS(y_train, X_train).fit()
# fit_regularized(alpha=0.2, L1_wt=0.5)
test0 = np.array([[1], [22], [60]]).T
score_values_predicted = model.predict(X_test)

rsquared_OLS = r2_score(y_test, score_values_predicted)

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)

params = np.append(clf.intercept_, clf.coef_)
predictions = clf.predict(X_test)

test = np.array([[22], [45], [52]]).T
# newX = pd.concat([pd.DataFrame({"Constant": np.ones(len(X_test))}), X_test], axis=1)
newX = pd.DataFrame({"Constant": np.ones(len(X_test))}).join(X_test.reset_index(drop=True))
# MSE = (np.sum((y_test.T-predictions.T)**2))/(len(newX)-len(newX.columns))
MSE = (sum((y_test.T-predictions.T)**2))/(len(newX)-len(newX.columns))

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


rsquared = r2_score(y_test, predictions)
F = rsquared / ((1 - rsquared) / (len(y) - 1 - 1))
p_value = stats.f.sf(F, 1, len(y) - 1 - 1)

# X = scaler_predictors.inverse_transform(X)
# y = scaler_output.inverse_transform(y)
# predictions = scaler_output.inverse_transform(predictions.reshape(-1, 1))

X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test.iloc[:, 0], predictions)
# Plot original data
plt.scatter(X_test.iloc[:, 0], y_test, color='red')
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
