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
