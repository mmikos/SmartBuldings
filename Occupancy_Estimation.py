# Data manipulation
import json
import datetime

import matplotlib.pylab as pylab
# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score
import math
import statsmodels.api as sm
import numpy as np
import pandas as pd
import requests
from pandas.io.json import json_normalize
# Preprocessing
from pandas.io.json import json_normalize
from sklearn.metrics import r2_score
import seaborn as sns
# Models
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

from Artificial_Neural_Network import ANN
from Artificial_Neural_Network_Classification import ANN_classify
from Artificial_Neural_Network_Regression import ANN_regress
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


def plot_confusion_matrix(confusion_matrix,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    classes = np.unique(y_train)
    if normalize:
        confusion_matrix = np.round(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], 2)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
end_date = "2019-11-24 17:45"

# AVUITY

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

# devices_list = pd.DataFrame([['OLY-A-413', 'OLY-A-414', 'OLY-A-415', 'OLY-A-416', 'OLY-A-417'],
#                              ['9033', '9049', '8989', '7675', '7663']])
devices_list = pd.DataFrame([['OLY-A-415'],
                             ['8989']])

awair_dataset = pd.DataFrame(columns=['comp', 'value', 'timestamp', 'score'])

for device_id in devices_list.iloc[1, :]:
    response_awair = requests.get(f'http://developer-apis.awair.is/v1/orgs/1097/devices/awair-omni/{device_id}/air-data'
                                  f'/15-min-avg', headers=headers, params=params)
    awair_json = response_awair.json()
    awair_str = json.dumps(awair_json)
    awair_data_dict = json.loads(awair_str)
    awair = pd.io.json.json_normalize(awair_data_dict["data"], record_path="sensors", meta=['timestamp', 'score'])
    awair_sensor_code = [device_id for _ in range(len(awair))]
    awair['sensor_name'] = awair_sensor_code
    awair_dataset = awair_dataset.append(awair, ignore_index=True)

awair = awair_dataset
# print(response_awair.status_code)

# rooms = ['ROOM 1', 'ROOM 2', 'ROOM 3', 'ROOM 4', 'ROOM 5']
rooms = ['ROOM 3']

occupancy_selected2 = pd.DataFrame(columns=['occupancy'])

for room in rooms:
    occupancy_selected = occupancy.loc[occupancy['areaName'] == f'{room}', ['startTs', 'occupancy']]
    date_occupancy_selected = occupancy_selected.set_index('startTs')

    date_occupancy_selected.index = pd.to_datetime(date_occupancy_selected.index, utc=True)

    date_occupancy_agg = date_occupancy_selected.resample('15T').max().ffill().astype(int)

    occupancy_selected2 = occupancy_selected2.append(date_occupancy_agg.reset_index(), ignore_index=False)
# occupancy_selected = occupancy[['startTs', 'occupancy', 'areaName']]

occupancy_selected = occupancy_selected2.sort_values(by='startTs')

# read the dataset
# measurements = pd.read_csv('data_sets/19_20/OLY-A-415.csv', sep=',', decimal='.')
# co2 = measurements[['timestamp(Europe/Berlin)', 'co2', 'noise']]
# date_co2 = co2[(co2['timestamp(Europe/Berlin)'] >= '2019-11-19 09:00')
#                & (co2['timestamp(Europe/Berlin)'] <= '2019-11-21 17:45')]

awair = awair.sort_values(by='timestamp', ascending=True)

awair['timestamp'] = pd.to_datetime(awair['timestamp'], format='%Y-%m-%d %H:%M', utc=True)

co2 = awair.loc[awair['comp'] == 'co2', ['timestamp', 'value']]
noise = awair.loc[awair['comp'] == 'spl_a', ['timestamp', 'value']]

# date_co2 = co2[(co2['timestamp(Europe/Berlin)'] >= '2019-11-19 09:00')
#                & (co2['timestamp(Europe/Berlin)'] <= '2019-11-21 17:45')]
# pd.to_datetime(date_co2.iloc[:, 0], format='%d/%m/%Y %H:%M').dt.time

#
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time')
# ax1.set_ylabel('Occupancy', color=color)
#
# ax1.plot(occupancy_selected.reset_index(drop=True).iloc[:, 1],
#          occupancy_selected.reset_index(drop=True).iloc[:, 0], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('CO2', color=color)
#
# ax2.plot(co2.iloc[:, 0], co2.iloc[:, 1], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()
# plt.title(f'Occupancy vs CO_2')
# # plt.legend()
# # plt.show()
#
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time')
# ax1.set_ylabel('Occupancy', color=color)
#
# ax1.plot(occupancy_selected.reset_index(drop=True).iloc[:, 1],
#          occupancy_selected.reset_index(drop=True).iloc[:, 0], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('noise', color=color)
#
# ax2.plot(noise.iloc[:, 0], noise.iloc[:, 1], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()
# plt.title(f'Occupancy vs Noise')
# plt.legend()
# plt.show()


# outlier_detection = DBSCAN(eps = .2, metric = 'euclidean', min_samples = 5, n_jobs = -1)
outlier_detection = KMeans(n_clusters=6, random_state=0)

# sns.boxplot(x= occupancy_selected.reset_index(drop=True).iloc[:, 0])
# plt.show()

# z = np.abs(stats.zscore(reg_data_no_na[['co2']]))
# print(z)
# threshold = 3
# print(np.where(z > 3))

# reg_data_no_na = reg_data_no_na[(z < 3).all(axis=1)]

# scaler_sound = StandardScaler()
# scaler_co2 = StandardScaler()
#
# sound = scaler_sound.fit_transform(reg_data_no_na[['noise']])
# sound = reg_data_no_na[['timestamp(Europe/Berlin)']]
# co2 = scaler_co2.fit_transform(reg_data_no_na[['co2']])

outlier_set = np.column_stack((noise.iloc[:, 1], co2.iloc[:, 1]))

clusters = outlier_detection.fit_predict(co2[['value']])

# cmap = cm.get_cmap('Set1')
# outlier_set[:, 0] = scaler_sound.inverse_transform(outlier_set[:, 0])
# outlier_set[:, 1] = scaler_co2.inverse_transform(outlier_set[:, 1])
# plt.scatter(outlier_set[:, 0], outlier_set[:, 1], c=clusters, cmap='viridis')
# plt.show()

# reg_data_no_na = reg_data_no_na - np.mean(reg_data_no_na)

# scaler_predictors = StandardScaler()
# scaler_output = StandardScaler()
co2_date = co2.set_index('timestamp')
noise_date = noise.set_index('timestamp')
occupancy_selected_date = occupancy_selected.set_index('startTs')

co2_date = co2_date.between_time('9:15', '16:45')
noise_date = noise_date.between_time('9:15', '16:45')
date_occupancy_agg = occupancy_selected_date.between_time('9:15', '16:45')

# measurements_labels = np.column_stack((co2_date, noise_date, date_occupancy_agg))
measurements = co2_date.join(noise_date, how='left', lsuffix='_co2', rsuffix='_noise')

measurements_labels = measurements.join(date_occupancy_agg, how='left')

X = measurements
y = date_occupancy_agg
data = measurements_labels

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# measurements = pd.DataFrame(np.column_stack((measurements, y)))

# plt.figure(figsize=(12,8))
# sns.heatmap(measurements.corr(), cmap='Blues',annot=True)
# plt.show()

# scaler = StandardScaler()
#
# X = scaler.fit_transform(X)
# X, y = shuffle(X, y)

# sensor_values_centered = sensor_values - np.mean(sensor_values)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

# clf = GaussianProcessClassifier(1.0 * RBF(1.0))
# clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50), shuffle=True, activation='relu', solver='lbfgs', alpha=0.0001,
#                     max_iter=5000)
# # clf = RandomForestClassifier(n_estimators=300, max_depth=10, criterion = 'entropy', class_weight='balanced', random_state = 42)
# # clf = SVC(kernel="linear", C=0.025)
# # clf = SVC(gamma=2, C=1)
#
# clf.fit(X_train, y_train.astype('int'))
#
# Random_Forest_class_predicition = clf.predict(X_test)
# print(accuracy_score(Random_Forest_class_predicition, y_test.astype('int')))
# print(precision_score(Random_Forest_class_predicition, y_test.astype('int'), average='weighted'))
# But Confusion Matrix and Classification Report give more details about performance
# print(confusion_matrix(Random_Forest_class_predicition, y_test))

# plt.hist(date_occupancy_agg.reset_index(drop=True).iloc[:, 0], bins='auto', color='lightblue', rwidth=0.85)
# plt.show()

number_of_nodes = 50
dropout_rate = 0.
number_of_epochs = 1000
regularization_penalty = 0.
window_size = 8
batch_size = 32
Results_ANN_list = []
# scores_test, pred_test, confusion_matrix = ANN_classify(X_train, X_test, y_train, y_test, number_of_nodes,
#                                                         number_of_layers, dropout_rate, number_of_epochs,
#                                                         regularization_penalty)

R2_ANN, MSE_ANN, y_predicted, training_history = ANN_regress(data, number_of_nodes, dropout_rate, window_size,
                                                             batch_size, number_of_epochs, regularization_penalty)

# for window in tqdm(window_size):
#     for batch in batch_size:
#
#         R2_ANN, MSE_ANN, y_predicted, training_history = ANN_regress(data, number_of_nodes, dropout_rate, window,
#                                                              batch, number_of_epochs, regularization_penalty)
#         Results_ANN_list.append([R2_ANN, MSE_ANN, window, batch])

# Results_ANN = pd.DataFrame(Results_ANN_list, columns=['R2', 'MSE', 'window_size', 'batch_size'])
# Results_ANN = Results_ANN.sort_values(['R2'], ascending=False)

plt.plot(training_history.history['mean_squared_error'])
plt.plot(training_history.history['val_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# number_of_nodes = [10, 50, 100]
# number_of_layers = [2, 4]
# dropout_rate = [0.1, 0.2]
# number_of_epochs = [1000, 2000]
# regularization_penalty = [0.001, 0.0001, 0.00001]
# Results_ANN_list = []
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, shuffle=True)
# # Fit ANN model
# plt.figure(figsize=(12, 10))
#
# for nodes in tqdm(number_of_nodes):
#     for layers in number_of_layers:
#         for dropout in dropout_rate:
#             for epochs in number_of_epochs:
#                 for regularization in regularization_penalty:
#                     scores_test, pred_test, confusion_matrix = ANN_classify(X_train, X_test, y_train, y_test, nodes, layers, dropout, epochs,
#                                                                             regularization)
#                     Results_ANN_list.append([nodes, layers, dropout, epochs, regularization, scores_test,
#                                              confusion_matrix])
#
# # plt.show()
# Results_ANN = pd.DataFrame(Results_ANN_list, columns=['Number of nodes', 'Number of layers', 'Dropout rate',
#                                                       'Number of epochs', 'Regularization penalty', 'scores_test',
#                                                       'confusion_matrix'])
# Results_ANN.sort_values(['scores_test'], ascending=False)

# Plot normalized confusion matrix
# plot_confusion_matrix(confusion_matrix, normalize=True, title='Normalized confusion matrix')

# plt.show()
degree = 3
# Perform polynomial transformation
polynomial_features = PolynomialFeatures(degree=degree)
sensor_values_polynomial = polynomial_features.fit_transform(X_train)

# Fit the model
model = sm.OLS(y_train, sensor_values_polynomial).fit()
# fit_regularized(alpha=0.2, L1_wt=0.5)
test0 = np.array([[1], [22], [60]]).T
score_values_predicted = model.predict(polynomial_features.fit_transform(X_test))

rsquared = model.rsquared_adj
MSE = math.sqrt(mean_squared_error(y_test, score_values_predicted))

print(model.summary())
X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, score_values_predicted)
# Plot original data
plt.scatter(X_test, y_test, color='red')
# plt.scatter(X[:, 1], y, color='blue')

plt.plot(X_test_sorted, y_predicted_sorted,
         label=f"Degree {1}," + f" $R^2$: {round(rsquared, 3)}, MSE: {round(MSE, 3)}")

plt.legend(loc='upper right')
plt.xlabel("CO_2")
plt.ylabel("Occupancy")
plt.title(f'Satisfaction vs indoor conditions (Polynomial Regression)')
plt.show()

print('whatever')

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
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)
#
# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(X_train, y_train)
#
# params = np.append(clf.intercept_, clf.coef_)
# predictions = clf.predict(X_test)
#
# # test = np.array([[22], [45], [52]]).T
# # newX = pd.concat([pd.DataFrame({"Constant": np.ones(len(X_test))}), X_test], axis=1)
# newX = pd.DataFrame({"Constant": np.ones(len(X_test))}).join(pd.DataFrame(X_test))
# # MSE = (np.sum((y_test.T-predictions.T)**2))/(len(newX)-len(newX.columns))
# MSE = (sum(((y_test.reset_index(drop = True).T-predictions.T))**2))/(len(newX)-len(newX.columns))
#
# var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
# sd_b = np.sqrt(var_b)
# ts_b = params/sd_b
#
# p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
#
# sd_b = np.round(sd_b, 3)
# ts_b = np.round(ts_b, 3)
# p_values = np.round(p_values, 3)
# params = np.round(params, 4)
#
# myDF3 = pd.DataFrame()
# myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
#                                                                                              p_values]
# print(myDF3)
#
#
# rsquared = r2_score(y_test, predictions)
# F = rsquared / ((1 - rsquared) / (len(y) - 1 - 1))
# p_value = stats.f.sf(F, 1, len(y) - 1 - 1)
#
# # X = scaler_predictors.inverse_transform(X)
# # y = scaler_output.inverse_transform(y)
# # predictions = scaler_output.inverse_transform(predictions.reshape(-1, 1))
