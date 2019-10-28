# Data manipulation
import numpy as np
import pandas as pd
import math

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from sklearn.decomposition import PCA

import KPI

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Models
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('raw_data_sensor.csv', sep=',', decimal='.')
# data = pd.read_csv('data_measurement.csv')

# data = data[['Hour', 'SpaceName', 'PortName', 'Value']]
data = data[
    ['Datetime', 'Date', 'Time', 'SpaceName', 'SpaceSubType', 'DeviceVendor', 'DeviceName', 'PortName', 'Value']]

mapping = pd.read_csv('mapping.csv', sep=',', decimal='.')

measurements = data['PortName'].unique()


# Function to get the measurements as seperate arrays - left for later

def get_single_measurement(data, measurement = str):

    measurement_name = data.loc[data['PortName'] == f'{measurement}', ['Datetime', 'SpaceName', 'DeviceName', 'PortName', 'Value']].pivot_table(index = ['Datetime'], columns = ['DeviceName'], values = ['Value'])
    measurement_name = measurement_name.fillna(0)
    return measurement_name



Temperature = get_single_measurement(data, 'Temperature')
Humidity = get_single_measurement(data, 'Humidity')
Light = get_single_measurement(data, 'Light')
Noise = get_single_measurement(data, 'Sound')
Occupancy = get_single_measurement(data, 'Occupancy')
Movement = get_single_measurement(data, 'Motion')
CO2 = get_single_measurement(data, 'CarbonDioxide')
Energy = get_single_measurement(data, 'EnergyConsumption')
Water = get_single_measurement(data, 'Volume')

# Temperature = Temperature.join(mapping.set_index('DeviceName'), on = 'DeviceName')

# normalise data - remove scale effect
measure = Temperature
centered_data = measure - np.mean(measure)

normalised_data = StandardScaler().fit_transform(centered_data)
# normalised_data = preprocessing.normalize(centered_data)
# normalised_data = centered_data
n_PC = 5
# pca = PCA(n_components = k)

model = PCA(n_components = n_PC).fit(normalised_data)
PCA_fit = model.transform(normalised_data)

PCA_components = pd.DataFrame(data = PCA_fit[:, :2], columns = ['PC1', 'PC2'])

# PCA_fit = pca.fit_transform(normalised_data)

# In case want to remove a column before and add it after PCA
# PCA_final = pd.concat([PCA, measure[['SpaceName']]])

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(1,1,1)

plt.plot(model.explained_variance_ratio_, '--o', label = 'Explained variance ratio')
plt.plot(model.explained_variance_ratio_.cumsum(), '--o', label = 'Cumulative explained variance ratio');

# ax.yaxis.set_major_formatter(PercentFormatter())

plt.yticks(np.arange(min(model.explained_variance_ratio_), max(model.explained_variance_ratio_) + 0.7, 0.1))
plt.legend()
plt.show()

n = len(normalised_data)

cov = np.matmul(normalised_data.T, normalised_data) / n

u, s, v = np.linalg.svd(cov)

k = 5

eigenvectors = u[:, 0:k]
eigenvalues = s[0:k]

scores = np.matmul(normalised_data, eigenvectors)

# projection of data onto k dimensional space
projection_of_data = np.matmul(scores, eigenvectors.T)

# projection onto residual space
projection_residual = np.matmul(normalised_data,
                                (np.diag(np.ones(len(eigenvectors))) - np.matmul(eigenvectors, eigenvectors.T)))

Q = []
T2 = []
# SPE = []

for row in range(len(projection_residual)):
    q = np.matmul(projection_residual[row], projection_residual[row].T)
    t = np.matmul(np.matmul(scores[row], np.linalg.inv(np.diag(eigenvalues))), scores[row].T)
    #     spe = np.linalg.norm(projection_residual[row])

    Q.append(q)
    T2.append(t)
#     SPE.append(spe)

confidence = 0.95
from scipy.stats import sem, t
from scipy import mean

n = len(normalised_data)

m = mean(Q)
std_err = sem(Q)
h = std_err * t.ppf((1 + confidence) / 2, k - 1)

conf_level_Q = (m + h) * np.ones(n)

from scipy.stats import chi2
# confidence level
p = 0.95
# chi2 value
# number of degrees of freedom is the number of k-most important PCs
valueChi2 = chi2.ppf(p, k)

conf_level_Chi = np.ones(n) * valueChi2

fig = plt.figure(figsize=(18, 16))

# plt.plot(Q, label = 'Q')
plt.plot(T2, label = 'T2')
# plt.plot(SPE, label = 'SPE')
# plt.plot(normalised_data[:, 0], label = 'CO2')

plt.plot(conf_level_Chi, label = 'Level of confidence T2')
# plt.plot(conf_level_Q, label = 'Level of confidence Q')

plt.legend()
plt.show()

anomaly = np.where(T2 > conf_level_Chi)

data_anomaly = measure.ix[anomaly]
data_anomaly.to_csv("anomaly.csv",  sep=';', decimal = ',')

print(Temperature)
