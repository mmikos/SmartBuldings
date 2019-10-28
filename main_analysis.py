# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from sklearn.decomposition import PCA
import statsmodels

# from statsmodels.multivariate.pca import PCA

plt.style.use('seaborn-whitegrid')

import KPI

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

import warnings

warnings.filterwarnings('ignore')

def get_single_measurement(data, measurement = str):
    measurement_name = data.loc[data['PortName'] == f'{measurement}', ['Hour', 'PortName', 'Value']].pivot_table(
        index='Hour', columns='PortName', values='Value')

    return measurement_name


#
# def get_single_measurement(data):
#
#     concatenated = pd.DataFrame()
#
#     list_of_measurements = data.PortName.unique()
#
#     for measure in list_of_measurements:
#         measurement_name = data.loc[data['PortName'] == measure, ['Hour', 'PortName', 'Value']].pivot(columns='PortName', values='Value')
#         concatenated = pd.concat([concatenated, measurement_name], axis=1)
#     return list


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

data = pd.read_csv('data_measurements_september.csv', sep=';', decimal='.')
# data = pd.read_csv('data_measurement.csv')

# data = data[['Hour', 'SpaceName', 'PortName', 'Value']]
data = data[['Datetime', 'Date', 'Hour', 'SpaceSubType', 'SpaceFriendlyName', 'PortName', 'Value']]

train, test = train_test_split(data, test_size=0.33, random_state=42)

# data = data['Value']['Temperature']

data_pivot = data.pivot_table(index='Datetime', columns='PortName', values='Value', aggfunc=np.mean)

measurements = data_pivot.columns.values

list_of_measurements = data.PortName.unique()

Movement = get_single_measurement(data, 'Motion')

Temperature = get_single_measurement(data, 'Temperature')
Humidity = get_single_measurement(data, 'Humidity')
Light = get_single_measurement(data, 'Light')
Noise = get_single_measurement(data, 'Sound')
Occupancy = get_single_measurement(data, 'Occupancy')
# Movement = get_single_measurement(data, 'Motion')
Movement = get_single_measurement(data, 'Movement')
CO2 = get_single_measurement(data, 'CO2')
# CO2 = get_single_measurement(data, 'CarbonDioxide')

# pivot the table so the measurements are in sepearate columns

data_pivot = data.pivot_table(index='Datetime', columns='PortName', values='Value')

measurements = data_pivot.columns.values

# Show summary statistics

data_pivot.describe()

# Plot correlation matrix

plt.figure(figsize=(12, 8))
sns.heatmap(data_pivot.corr(), cmap='Blues', annot=True)

# Outliers analysis basen on box blot

no_of_columns = len(measurements)
plt.figure(figsize=(no_of_columns, 5 * no_of_columns))

for i in range(0, len(measurements)):
    plt.subplot(no_of_columns + 1, no_of_columns, i + 1)
    sns.boxplot(data_pivot[measurements[i]], color='lightblue', orient='v')
    plt.tight_layout()

# Analyze distribution and skewness

data_pivot_no_nan = data_pivot.dropna()
plt.figure(figsize=(2 * no_of_columns, 5 * no_of_columns))
for i in range(0, len(measurements)):
    plt.subplot(no_of_columns + 1, no_of_columns, i + 1)
    sns.distplot(data_pivot_no_nan[measurements[i]], kde=True)

# See the missing values

missingno.matrix(data_pivot, figsize=(30, 10))

# data_centered = data_pivot_no_nan - np.mean(data_pivot_no_nan)

normalised_data = StandardScaler().fit_transform(data_pivot_no_nan)

k = 5
# pca = statsmodels.multivariate.pca.PCA(normalised_data)
# fig = pca.plot_rsquare(ncomp=4)
# pca = PCA(n_components = k)
#
# PCA_fit = pca.fit_transform(normalised_data)
#
n_PC = 10
model = PCA(n_components = n_PC).fit(normalised_data)
PCA_fit = model.transform(normalised_data)

PCA_components = pd.DataFrame(data = PCA_fit[:, :5], columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])


# important_features = [np.abs(model.components_[i]) for i in range(n_PC)]
#
# measurement_names = [measurements[important_features[i]] for i in range(n_PC)]
#
# # create table
# dic = {'PC {}'.format(i): measurement_names[i] for i in range(n_PC)}
#
# df = pd.DataFrame(dic.items())

n = len(normalised_data)

cov = np.matmul(normalised_data.T, normalised_data)/n

u, s, v = np.linalg.svd(cov)

# P matrix
eigenvectors = u[:, 0:k]
eigenvalues = s[0:k]

# np.matmul(eigenvectors.T, normalised_data.T)

# T matrix
scores = np.matmul(normalised_data, eigenvectors)

#projection of data onto k dimensional space
projection_of_data = np.matmul(scores, eigenvectors.T)
# projection_of_data2 = np.matmul(normalised_data, np.matmul(eigenvectors, eigenvectors.T))

#projection onto residual space
projection_residual = np.matmul(normalised_data, (np.diag(np.ones(len(eigenvectors))) - np.matmul(eigenvectors, eigenvectors.T)))

Q = []
T2 = []

for row in range(len(projection_residual)):

    q = np.matmul(projection_residual[row], projection_residual[row].T)
    t = np.matmul(np.matmul(scores[row], np.linalg.inv(np.diag(eigenvalues))), scores[row].T)

    Q.append(q)
    T2.append(t)

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
print(valueChi2)

fig = plt.figure(figsize=(18, 16))

plt.plot(Q, label = 'Q')
plt.plot(T2, label = 'T2')
# plt.plot(SPE, label = 'SPE')
plt.plot(normalised_data[:, 2], label = 'CO2')

conf_level_Chi = np.ones(n) * valueChi2

plt.plot(conf_level_Chi, label = 'Level of confidence T2')
plt.plot(conf_level_Q, label = 'Level of confidence Q')

plt.legend()
plt.show()


residual_matrix = normalised_data - projection_of_data

original_data = projection_of_data + residual_matrix

singa = np.linalg.inv(np.diag(s[0:k]))

T2 = np.matmul(np.matmul(np.matmul(np.matmul(normalised_data, eigenvectors), singa), eigenvectors.T), normalised_data.T)

spe = np.linalg.norm((np.matmul(np.matmul(u[:, k+1:], u[:, k+1:].T), normalised_data.T)))

# In case want to remove a column before and add it after PCA
# PCA_final = pd.concat([PCA_components, measurements_names], axis=1)
np.diag(np.diag(eigenvectors))
fig = plt.figure(figsize=(10, 10))

plt.plot(normalised_data[:, 0], label='Original data')
plt.plot(PCA_components.iloc[:, 0], label='PC 1')
plt.plot(PCA_components.iloc[:, 1], label='PC 2')
plt.plot(PCA_components.iloc[:, 2], label='PC 3')
plt.plot(PCA_components.iloc[:, 3], label='PC 4')
plt.plot(PCA_components.iloc[:, 4], label='PC 5')

plt.legend()

plt.show()

print(model.explained_variance_ratio_)

print(model.singular_values_)

# KPI calculation

# data_pivot_CO2 = KPI.caclulate_KPIs('CO2', data_pivot)


print(list_of_measurements)

# Temperature['Hour'] = pd.to_datetime(Temperature['Hour'], format = "%d.%m.%Y %H:%M:%S")
