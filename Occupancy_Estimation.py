# Data manipulation
import numpy as np
import pandas as pd
import math

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Models
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

# read the dataset
measure = pd.read_csv('data_sets/conference_room_sensor_data_dev.csv', sep=',', decimal='.')
# measure = pd.read_csv('data_sets/OLY-A-415.csv', sep=',', decimal='.')

def get_single_measurement(data, measurement=str):
    measurement_name = data.loc[data['PortName'] == f'{measurement}', ['Datetime', 'PortName', 'Value']].pivot_table(
        index='Datetime', columns='PortName', values='Value')

    return measurement_name


Temperature = get_single_measurement(measure, 'Temperature')
Humidity = get_single_measurement(measure, 'Humidity')
Light = get_single_measurement(measure, 'Light')
Noise = get_single_measurement(measure, 'Sound')
Occupancy = get_single_measurement(measure, 'Occupancy')
Movement = get_single_measurement(measure, 'Motion')
CO2 = get_single_measurement(measure, 'CarbonDioxide')

print('whatever')