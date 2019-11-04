import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

os.chdir("Parsing/Temperature")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# d = pd.read_csv('2018-06.csv', delimiter=";")
# value = d[['Value']]
# nvalue = value.to_numpy()
# average = np.mean(nvalue)
# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, decimal=',', delimiter=";") for f in all_filenames])
# combined_csv = pd.concat([pd.read_csv(f, delimiter=";") for f in all_filenames])
#
# lower_filter = combined_csv[combined_csv['Value'] >= 0 ]
# upper_filter = lower_filter[lower_filter['Value'] < 2500 ]

lower_filter = combined_csv[combined_csv['Value'] >= 16 ]
upper_filter = lower_filter[lower_filter['Value'] < 31 ]

upper_filter = upper_filter[['Value']]

average = np.mean(upper_filter)

upper_filter.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

# value = combined_csv[['Value']]
#
# nvalue = value.to_numpy()
#
# filtered = nvalue[~np.isnan(nvalue).any(axis=1)]
# filtered = filtered[ (filtered > 0) & (filtered < 2500) ]
#
# average = np.mean(filtered)
#
# df = pd.DataFrame(filtered)
# # export to csv
# df.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')


#
#
#
#
#
# print(average)
