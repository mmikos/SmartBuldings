import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

data = pd.read_csv('CO2data.csv')
data = data[data["SpaceFriendlyName"] == 'C-Level']
# data_array = data.to_numpy()
#
# values = data_array[:, 1]
# date_time = data_array[:, 2]
values = data["value"]
date_time = data["date_time"]

corr = data.loc[data['DataTypeName'].isin(['CO2', 'Occupancy', 'Movement', 'Light'])]
corr2 = corr[['DataTypeName', 'value']]

pivot = corr2.pivot(columns='DataTypeName', values='value')
pivot.to_csv(r'pivot_data.csv')

correlation = pivot.corr()

# plot the heatmap
sns.heatmap(correlation,
        xticklabels=correlation.columns,
        yticklabels=correlation.columns)

plt.show()
subspaces = data["SpaceFriendlyName"].unique()

for subspace in subspaces:
    plot_data = data[data["SpaceFriendlyName"] == subspace]
    CO2 = plot_data[plot_data["DataTypeName"] == "CO2"]
    occupancy = plot_data[plot_data["DataTypeName"] == "Occupancy"]
    plt.plot(CO2["date_time"], CO2["value"], label="CO2")
    plt.plot(occupancy["date_time"], occupancy["value"], label="Occupancy")
    # plt.ylim(400, 800)
    # plt.xlim(0, 24)
    plt.legend()
    plt.title('Name of the space: ' + str(subspace))
    plt.show()
# plt.plot(date_time, values)
