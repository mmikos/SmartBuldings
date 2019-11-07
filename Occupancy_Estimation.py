# Data manipulation
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns

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

room_5 = pd.read_csv('data_sets/Room_55.csv', sep=';', decimal='.')

co2 = measure[['timestamp(Europe/Berlin)', 'co2']]

date_co2 = co2[(co2['timestamp(Europe/Berlin)'] >= '29/10/2019') & (co2['timestamp(Europe/Berlin)'] <= '31/10/2019')]

no_occupants = room_5[['date_time', 'no_occupants']]
# measure = pd.read_csv('data_sets/OLY-A-415.csv', sep=',', decimal='.')
date_no_occupants = no_occupants[(no_occupants['date_time'] >= '29/10/2019') & (no_occupants['date_time'] <= '31/10/2019')]

date_no_occupants = date_no_occupants.sort_values(by = ['date_time'])
date_co2 = date_co2.sort_values(by = ['timestamp(Europe/Berlin)'])

# plt.figure(figsize=(12, 10))
# plt.plot(date_no_occupants.iloc[:, 0], date_no_occupants.iloc[:, 1], label = 'Occupancy')
# plt.legend()
# plt.show()

# fig, axs = plt.subplots(2)
# fig.suptitle('Aligning x-axis using sharex')
# axs[0].plot(date_no_occupants.iloc[:, 0], date_no_occupants.iloc[:, 1])
# axs[1].plot(date_co2.iloc[:, 0], date_co2.iloc[:, 1])
# plt.show()

# plt.figure(figsize=(12, 10))
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time')
# ax1.set_ylabel('Occupancy', color=color)
#
# ax1.plot(date_no_occupants.iloc[:, 0], date_no_occupants.iloc[:, 1], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('CO2', color=color)
#
# ax2.plot(date_co2.iloc[:, 0], date_co2.iloc[:, 1], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()
# # plt.xlabel(f"{sensor_name}")
# # plt.ylabel(f"{score_name}")
# plt.title(f'Occupancy')
# plt.legend()
# plt.show()

data_bricks = pd.read_csv('data_sets/conference_room_sensor_data.csv', sep=',', decimal='.')


def get_single_measurement(data, measurement=str):
    measurement_name = data.loc[data['PortName'] == f'{measurement}', ['Datetime', 'PortName', 'Value']].pivot_table(
        index='Datetime', columns='PortName', values='Value')

    return measurement_name


Temperature = get_single_measurement(data_bricks, 'Temperature')
Humidity = get_single_measurement(data_bricks, 'Humidity')
Light = get_single_measurement(data_bricks, 'Light')
Noise = get_single_measurement(data_bricks, 'Sound')
Occupancy = get_single_measurement(data_bricks, 'Occupancy')
Movement = get_single_measurement(data_bricks, 'Motion')
CO2 = get_single_measurement(data_bricks, 'CarbonDioxide')

print('whatever')
