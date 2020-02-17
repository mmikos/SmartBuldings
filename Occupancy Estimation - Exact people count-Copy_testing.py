#!/usr/bin/env python
# coding: utf-8

# ## Regression models for occupancy estimation based on $CO_2$

# In[151]:


import math
import matplotlib.pylab as pylab
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from Artificial_Neural_Network_Classification import ANN_classify
from Artificial_Neural_Network_Regression import *
from Decision_Tree_Regression import Decision_Tree_Regression
from Get_Data_From_API import get_data_from_API
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium'}
pylab.rcParams.update(params)
import warnings

warnings.filterwarnings('ignore')
np.random.seed(48)

def plot_confusion_matrix(confusion_matrix_calculated,
                          accuracy,
                          model_name):
    matrix_percentage = confusion_matrix_calculated.astype(dtype=np.float32)

    rows = confusion_matrix_calculated.shape[0]
    columns = confusion_matrix_calculated.shape[1]

    for column in range(columns):
        for row in range(rows):
            matrix_percentage[row, column] = matrix_percentage[row, column] /                                              np.sum(confusion_matrix_calculated, axis=1)[row]

    plt.figure(figsize=(9, 9))
    sns.heatmap(matrix_percentage, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = f'Accuracy Score for {model_name}: {np.round(accuracy, 4)*100} %'
    plt.title(all_sample_title, size=15)
    plt.show()


def filter_measurements_data(measure, name, business_hours, weekends):
    
    if name == 'co2':
        measure = measure.shift(1)
        measure = measure.fillna(method='bfill')
        
    if business_hours:
        measure = measure.between_time('8:00', '19:00')

    if weekends:
        measure = measure[measure.index.dayofweek < 5]

    measure = measure.rename(columns={'value': f'{name}'})

    return measure


def create_dataset(occupancy_data, *args):
    data_set = pd.DataFrame()
    measurements = occupancy_data

    for ar in args:
        data_set = measurements.join(ar, how='inner')
        measurements = data_set

    return data_set


def prepare_data(data_set, size_window):
    batches_list = []
    labels_list = []
    features_number = data_set.shape[1] - 1
    for idx in range(len(data_set) - size_window - 1):
        batches_list.append(data_set[idx: idx + size_window, 1:features_number + 1])
        labels_list.append(data_set[idx + size_window, 0])
    return np.array(batches_list), np.array(labels_list)


# ## Occupancy estimation
# 
# Two models will be applied to the problem of occupancy estimation:
# 
# * Classification problem - Room is occupied or not
# * Regression problem - How many people are in the room

# In[152]:


data = pd.read_csv('data_co2_occ_full_numerical.csv', sep=';', decimal=',', index_col=0)
data.index = pd.to_datetime(data.index, utc=True)


# In[153]:


# account for the lag in co2 increase and people entering the room

data['co2'] = data['co2'].shift(1)
data['co2'] = data['co2'].fillna(method='bfill')

data['noise'] = data['noise'].shift(2)
data['noise'] = data['noise'].fillna(method='bfill')


# In[154]:


data_co2_room1 = data.loc[data['room'] == 1, ['occupancy', 'co2']]
data_co2_room2 = data.loc[data['room'] == 2, ['occupancy', 'co2']]
data_co2_room3 = data.loc[data['room'] == 3, ['occupancy', 'co2']]
data_co2_room4 = data.loc[data['room'] == 4, ['occupancy', 'co2']]
data_co2_room5 = data.loc[data['room'] == 5, ['occupancy', 'co2']]

data_co2_noise_room1 = data.loc[data['room'] == 1, ['occupancy', 'co2', 'noise']]
data_co2_noise_room2 = data.loc[data['room'] == 2, ['occupancy', 'co2', 'noise']]
data_co2_noise_room3 = data.loc[data['room'] == 3, ['occupancy', 'co2', 'noise']]
data_co2_noise_room4 = data.loc[data['room'] == 4, ['occupancy', 'co2', 'noise']]
data_co2_noise_room5 = data.loc[data['room'] == 5, ['occupancy', 'co2', 'noise']]

data_co2_noise_humidity_temperature_room4 = data.loc[data['room'] == 4, ['occupancy', 'co2', 'noise', 'humidity', 'temperature']]

data_co2_full = data[['occupancy', 'co2']]
data_co2_noise_full = data[['occupancy', 'co2', 'noise']]


# In[155]:


plt.figure(figsize=(12,8))
sns.heatmap(data_co2_noise_humidity_temperature_room4.corr(),cmap='Blues',annot=True)


# ### Visualize relationship for 2 day
# 
# 1 - room occupied
# 0 - room not occupied
# 

# In[156]:


data_co2_room = data_co2_room4

data_cut = (data_co2_room.loc[(data_co2_room.index.day==25) & (data_co2_room.index.month==11)]).append(data_co2_room.loc[(data_co2_room.index.day==26) & (data_co2_room.index.month==11)])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(data_cut['occupancy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CO_2', color=color)

ax2.plot(data_cut['co2'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs CO_2')


# In[157]:


data_noise_room = data_co2_noise_room4

data_cut = (data_noise_room.loc[(data_noise_room.index.day==25) & (data_noise_room.index.month==11)]).append(data_noise_room.loc[(data_noise_room.index.day==26) & (data_noise_room.index.month==11)])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(data_cut['occupancy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Sound', color=color)

ax2.plot(data_cut['noise'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs Sound')

#
# ax2.set_ylabel('Noise', color=color)
#
# ax2.plot(data['noise'], color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()
# plt.title(f'Occupancy vs Noise')


# ## How imbalanced is the dataset?

# In[158]:


count_class = data_co2_room4['occupancy'].value_counts()
count_class


# In[159]:


plt.bar(count_class.index, [count_class.loc[0], count_class.loc[1], count_class.loc[2], count_class.loc[3],
                            count_class.loc[4], count_class.loc[5]], width=0.2, align='center', alpha=0.5)

plt.ylabel('Number of samples')
plt.title('Dataset imbalance visualization')


# Scale and normalize feature data

# In[160]:


data = data_co2_room4
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])


# ## Estimation models 

# In[161]:


max_depth = 6
n_estimators = 150

R2_DT, MSE_DT, prediction_DT, y_test_DT = Decision_Tree_Regression(data[:, 1:no_features + 1].reshape(-1, no_features),
                                                        data[:, 0].reshape(-1, 1), max_depth, n_estimators, 'plot', 'measure', 'occ')

prediction_DT = np.round(prediction_DT, 0)

fitness_DT = prediction_DT.T == y_test_DT.T
good_pred = np.sum(fitness_DT)
bad_pred = fitness_DT.shape[1] - np.sum(fitness_DT)

exact_accuracy_DT = good_pred / (good_pred + bad_pred)

pd.DataFrame([R2_DT, MSE_DT, exact_accuracy_DT], columns=['Evaluation'], index=['R2', 'MSE', 'Accuarcy'])


# In[162]:


X_train_SVR, X_test_SVR, y_train_SVR, y_test_SVR = train_test_split(data[:, 1:no_features + 1], data[:, 0],
                                                                    test_size=1 / 3, random_state=42, shuffle=True)

model_SVR = SVR(C=1.0, epsilon=0.2)

features_number = X_train_SVR.shape[1]

y_train_SVR = y_train_SVR.astype('int')
y_test_SVR = y_test_SVR.astype('int')

model_SVR.fit(X_train_SVR.reshape(-1, features_number), y_train_SVR.reshape(-1, 1))
prediction_SVR = model_SVR.predict(X_test_SVR.reshape(-1, features_number))
R2_SVR = model_SVR.score(X_test_SVR, y_test_SVR)
MSE_SVR = math.sqrt(mean_squared_error(y_test_SVR, prediction_SVR))

exact_accuracy_SVR = good_pred / (good_pred + bad_pred)
pd.DataFrame([R2_SVR, MSE_SVR], columns=['Evaluation'], index=['R2', 'MSE'])


# In[163]:


number_of_epochs = 1500
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[164]:


number_of_nodes_MLP = 64
model_MLP = MLP(window_size, no_features, number_of_nodes_MLP)
training_history_MLP = model_MLP.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_MLP = model_MLP.predict(X_test)
MSE_MLP_co2_room4, R2_MLP_co2_room4, residuals_MLP_co2_room4, accuracy_MLP_co2_room4, mean_error_MLP_co2_room4 = model_MLP.evaluate(prediction_MLP, X_test, y_test)


# In[165]:


slice_start = 160
slice_end = 220

p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_MLP[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using MLP')


# In[166]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM_co2_room4 = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM_co2_room4 = model_LSTM.predict(X_test)
MSE_LSTM_co2_room4, R2_LSTM_co2_room4, residuals_LSTM_co2_room4, accuracy_LSTM_co2_room4, mean_error_LSTM_co2_room4 = model_LSTM.evaluate(prediction_LSTM_co2_room4, X_test, y_test)


# In[167]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM_co2_room4[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using LSTM')


# In[168]:


plt.figure(figsize=(12, 10))
plt.plot(training_history_LSTM_co2_room4.history['loss'])
plt.plot(training_history_LSTM_co2_room4.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[169]:


number_of_filters_CNN = 64
model_CNN = CNN(window_size, no_features, number_of_filters_CNN)
training_history_CNN = model_CNN.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_CNN = model_CNN.predict(X_test)
MSE_CNN_co2_room4, R2_CNN_co2_room4, residuals_CNN_co2_room4, accuracy_CNN_co2_room4, mean_error_CNN_co2_room4 = model_CNN.evaluate(prediction_CNN, X_test, y_test)


# In[170]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_CNN[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using CNN')


# In[171]:


number_of_filters_CNN_LSTM = 64
number_of_nodes_CNN_LSTM = 64
model_CNN_LSTM = CNN_LSTM(window_size, no_features, number_of_filters_CNN_LSTM, number_of_nodes_CNN_LSTM)
training_history_CNN_LSTM = model_CNN_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs)
prediction_CNN_LSTM = model_CNN_LSTM.predict(X_test)
MSE_CNN_LSTM_co2_room4, R2_CNN_LSTM_co2_room4, residuals_CNN_LSTM_co2_room4, accuracy_CNN_LSTM_co2_room4, mean_error_CNN_LSTM_co2_room4 = model_CNN_LSTM.evaluate(
    prediction_CNN_LSTM, X_test, y_test)


# In[172]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_CNN_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using CNN LSTM')


# ### Adding sound

# In[173]:


data = data_co2_noise_room4
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])


# In[174]:


number_of_epochs = 1500
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[ ]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM_co2_noise_room4 = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room4, R2_LSTM_co2_noise_room4, residuals_LSTM_co2_noise_room4, accuracy_LSTM_co2_noise_room4, mean_error_LSTM_co2_noise_room4 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[ ]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using CNN LSTM')


# In[ ]:


plt.figure(figsize=(12, 10))
plt.plot(training_history_LSTM_co2_noise_room4.history['loss'])
plt.plot(training_history_LSTM_co2_noise_room4.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[ ]:


pd.DataFrame([[MSE_MLP_co2_room4, R2_MLP_co2_room4, accuracy_MLP_co2_room4, mean_error_MLP_co2_room4],
              [MSE_LSTM_co2_room4, R2_LSTM_co2_room4, accuracy_LSTM_co2_room4, mean_error_LSTM_co2_room4],
              [MSE_CNN_co2_room4, R2_CNN_co2_room4, accuracy_CNN_co2_room4, mean_error_CNN_co2_room4],
              [MSE_CNN_LSTM_co2_room4, R2_CNN_LSTM_co2_room4, accuracy_CNN_LSTM_co2_room4, mean_error_CNN_LSTM_co2_room4], 
              [MSE_LSTM_co2_noise_room4, R2_LSTM_co2_noise_room4, accuracy_LSTM_co2_noise_room4, mean_error_LSTM_co2_noise_room4]],
             columns=['MSE', 'R2', 'accuracy', 'mean_error'], index=['MPL', 'LSTM', 'CNN', 'CNN_LSTM', 'LSTM + noise'])


# In[ ]:


X_test


# In[ ]:


model_LSTM.predict(X_test)

825 ->5
1234 ->4


# ## Room 1 
# ### $CO_2$ only

# In[ ]:


data = data_co2_room1
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 800
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[25]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_room1, R2_LSTM_co2_room1, residuals_LSTM_co2_room1, accuracy_LSTM_co2_room1, mean_error_LSTM_co2_room1 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# #### Adding sound

# In[26]:


data = data_co2_noise_room1
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[27]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room1, R2_LSTM_co2_noise_room1, residuals_LSTM_co2_noise_room1, accuracy_LSTM_co2_noise_room1, mean_error_LSTM_co2_noise_room1 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# ## Room 2
# ### $CO_2$ only

# In[28]:


data = data_co2_room2
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[29]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_room2, R2_LSTM_co2_room2, residuals_LSTM_co2_room2, accuracy_LSTM_co2_room2, mean_error_LSTM_co2_room2 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# #### Adding sound

# In[30]:


data = data_co2_noise_room2
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[31]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room2, R2_LSTM_co2_noise_room2, residuals_LSTM_co2_noise_room2, accuracy_LSTM_co2_noise_room2, mean_error_LSTM_co2_noise_room2 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# ## Room 3
# ### $CO_2$ only

# In[32]:


data = data_co2_room3
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[33]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_room3, R2_LSTM_co2_room3, residuals_LSTM_co2_room3, accuracy_LSTM_co2_room3, mean_error_LSTM_co2_room3 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# #### Adding sound

# In[34]:


data = data_co2_noise_room3
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[35]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room3, R2_LSTM_co2_noise_room3, residuals_LSTM_co2_noise_room3, accuracy_LSTM_co2_noise_room3, mean_error_LSTM_co2_noise_room3 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# ## Room 5
# ### $CO_2$ only

# In[36]:


data = data_co2_room5
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[37]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_room5, R2_LSTM_co2_room5, residuals_LSTM_co2_room5, accuracy_LSTM_co2_room5, mean_error_LSTM_co2_room5 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# #### Adding sound

# In[38]:


data = data_co2_noise_room5
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[39]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room5, R2_LSTM_co2_noise_room5, residuals_LSTM_co2_noise_room5, accuracy_LSTM_co2_noise_room5, mean_error_LSTM_co2_noise_room5 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# ## All rooms  together
# ### $CO_2$ only

# In[40]:


data = data_co2_full
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[41]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_all, R2_LSTM_co2_all, residuals_LSTM_co2_all, accuracy_LSTM_co2_all, mean_error_LSTM_co2_all = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[42]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using LSTM')


# In[43]:


number_of_filters_CNN_LSTM = 64
number_of_nodes_CNN_LSTM = 64
model_CNN_LSTM = CNN_LSTM(window_size, no_features, number_of_filters_CNN_LSTM, number_of_nodes_CNN_LSTM)
training_history_CNN_LSTM = model_CNN_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs)
prediction_CNN_LSTM = model_CNN_LSTM.predict(X_test)
MSE_CNN_LSTM_co2_all, R2_CNN_LSTM_co2_all, residuals_CNN_LSTM_co2_all, accuracy_CNN_LSTM_co2_all, mean_error_CNN_LSTM_co2_all = model_CNN_LSTM.evaluate(
    prediction_CNN_LSTM, X_test, y_test)


# In[44]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_CNN_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using CNN LSTM')


# In[45]:


pd.DataFrame([[MSE_LSTM_co2_all, R2_LSTM_co2_all, accuracy_LSTM_co2_all, mean_error_LSTM_co2_all],
              [MSE_CNN_LSTM_co2_all, R2_CNN_LSTM_co2_all, accuracy_CNN_LSTM_co2_all, mean_error_CNN_LSTM_co2_all]],
             columns=['MSE', 'R2', 'accuracy', 'mean_error'], index=['LSTM', 'CNN_LSTM'])


# #### Adding sound

# In[46]:


data = data_co2_noise_full
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1200
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[47]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_all, R2_LSTM_co2_noise_all, residuals_LSTM_co2_noise_all, accuracy_LSTM_co2_noise_all, mean_error_LSTM_co2_noise_all = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[48]:


pd.DataFrame([[MSE_LSTM_co2_room1, R2_LSTM_co2_room1, accuracy_LSTM_co2_room1, mean_error_LSTM_co2_room1],
              [MSE_LSTM_co2_room2, R2_LSTM_co2_room2, accuracy_LSTM_co2_room2, mean_error_LSTM_co2_room2],
              [MSE_LSTM_co2_room3, R2_LSTM_co2_room3, accuracy_LSTM_co2_room3, mean_error_LSTM_co2_room3],
              [MSE_LSTM_co2_room4, R2_LSTM_co2_room4, accuracy_LSTM_co2_room4, mean_error_LSTM_co2_room4],
              [MSE_LSTM_co2_room5, R2_LSTM_co2_room5, accuracy_LSTM_co2_room5, mean_error_LSTM_co2_room5],
              [MSE_LSTM_co2_all, R2_LSTM_co2_all, accuracy_LSTM_co2_all, mean_error_LSTM_co2_all]],
             columns=['MSE', 'R2', 'accuracy', 'mean_error'], index=['Room 1', 'Room 2', 'Room 3', 'Room 4', 'Room 5', 'All rooms'])


# In[49]:


pd.DataFrame([[MSE_LSTM_co2_noise_room1, R2_LSTM_co2_noise_room1, accuracy_LSTM_co2_noise_room1, mean_error_LSTM_co2_noise_room1],
              [MSE_LSTM_co2_noise_room2, R2_LSTM_co2_noise_room2, accuracy_LSTM_co2_noise_room2, mean_error_LSTM_co2_noise_room2],
              [MSE_LSTM_co2_noise_room3, R2_LSTM_co2_noise_room3, accuracy_LSTM_co2_noise_room3, mean_error_LSTM_co2_noise_room3],
              [MSE_LSTM_co2_noise_room4, R2_LSTM_co2_noise_room4, accuracy_LSTM_co2_noise_room4, mean_error_LSTM_co2_noise_room4],
              [MSE_LSTM_co2_noise_room5, R2_LSTM_co2_noise_room5, accuracy_LSTM_co2_noise_room5, mean_error_LSTM_co2_noise_room5],
              [MSE_LSTM_co2_noise_all, R2_LSTM_co2_noise_all, accuracy_LSTM_co2_noise_all, mean_error_LSTM_co2_noise_all]],
              columns=['MSE', 'R2', 'accuracy', 'mean_error'], index=['Room 1', 'Room 2', 'Room 3', 'Room 4', 'Room 5', 'All rooms'])


# # Change sample frequency to 15 min

# In[2]:


data = pd.read_csv('data_co2_occ_full_numerical_15.csv', sep=';', decimal=',', index_col=0)
data.index = pd.to_datetime(data.index, utc=True)

# account for the lag in co2 increase and people entering the room

data['co2'] = data['co2'].shift(3)
data['co2'] = data['co2'].fillna(method='bfill')

data['noise'] = data['noise'].shift(4)
data['noise'] = data['noise'].fillna(method='bfill')

data_co2_room1 = data.loc[data['room'] == 1, ['occupancy', 'co2']]
data_co2_room2 = data.loc[data['room'] == 2, ['occupancy', 'co2']]
data_co2_room3 = data.loc[data['room'] == 3, ['occupancy', 'co2']]
data_co2_room4 = data.loc[data['room'] == 4, ['occupancy', 'co2']]
data_co2_room5 = data.loc[data['room'] == 5, ['occupancy', 'co2']]

data_co2_noise_room1 = data.loc[data['room'] == 1, ['occupancy', 'co2', 'noise']]
data_co2_noise_room2 = data.loc[data['room'] == 2, ['occupancy', 'co2', 'noise']]
data_co2_noise_room3 = data.loc[data['room'] == 3, ['occupancy', 'co2', 'noise']]
data_co2_noise_room4 = data.loc[data['room'] == 4, ['occupancy', 'co2', 'noise']]
data_co2_noise_room5 = data.loc[data['room'] == 5, ['occupancy', 'co2', 'noise']]

data_co2_full = data[['occupancy', 'co2']]
data_co2_noise_full = data[['occupancy', 'co2', 'noise']]


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(data_co2_noise_humidity_temperature_room4.corr(),cmap='Blues',annot=True)


# In[3]:


data_co2_room = data_co2_room4

data_cut = (data_co2_room.loc[(data_co2_room.index.day==25) & (data_co2_room.index.month==11)]).append(data_co2_room.loc[(data_co2_room.index.day==26) & (data_co2_room.index.month==11)])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(data_cut['occupancy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CO_2', color=color)

ax2.plot(data_cut['co2'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs CO_2')


# In[11]:


data_noise_room = data_co2_noise_room4

data_cut = (data_noise_room.loc[(data_noise_room.index.day==25) & (data_noise_room.index.month==11)]).append(data_noise_room.loc[(data_noise_room.index.day==26) & (data_noise_room.index.month==11)])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Occupancy', color=color)

ax1.plot(data_cut['occupancy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Sound', color=color)

ax2.plot(data_cut['noise'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs Sound')


# In[12]:


data = data_co2_room4
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])


# In[13]:


number_of_epochs = 1000
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[14]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_room4_15, R2_LSTM_co2_room4_15, residuals_LSTM_co2_room4_15, accuracy_LSTM_co2_room4_15, mean_error_LSTM_co2_room4_15 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[15]:


slice_start = 160
slice_end = 220

p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using LSTM')


# ### Adding sound

# In[16]:


data = data_co2_noise_room4
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1000
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[17]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_room4_15, R2_LSTM_co2_noise_room4_15, residuals_LSTM_co2_noise_room4_15, accuracy_LSTM_co2_noise_room4_15, mean_error_LSTM_co2_noise_room4_15 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# ## All rooms  together
# ### $CO_2$ only

# In[18]:


data = data_co2_full
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1000
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[19]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_full_15, R2_LSTM_co2_full_15, residuals_LSTM_co2_full_15, accuracy_LSTM_co2_full_15, mean_error_LSTM_co2_full_15 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[20]:


slice_start = 160
slice_end = 220

p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using LSTM')


# ### Adding sound

# In[21]:


data = data_co2_noise_full
scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

number_of_epochs = 1000
window_size = 32
batch_size = 64

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)


# In[22]:


number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM_co2_noise_full_15, R2_LSTM_co2_noise_full_15, residuals_LSTM_co2_noise_full_15, accuracy_LSTM_co2_noise_full_15, mean_error_LSTM_co2_noise_full_15 = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)


# In[23]:


p1 = plt.plot(y_test[slice_start:slice_end])
p2 = plt.plot(prediction_LSTM[slice_start:slice_end])

plt.legend((p1[0], p2[0]), ('Real values', 'Prediction'))

plt.title('Real vs predicted using LSTM')


# In[25]:


pd.DataFrame([[MSE_LSTM_co2_room4_15, R2_LSTM_co2_room4_15, accuracy_LSTM_co2_room4_15, mean_error_LSTM_co2_room4_15], 
              [MSE_LSTM_co2_noise_room4_15, R2_LSTM_co2_noise_room4_15, accuracy_LSTM_co2_noise_room4_15, mean_error_LSTM_co2_noise_room4_15],
              [MSE_LSTM_co2_full_15, R2_LSTM_co2_full_15, accuracy_LSTM_co2_full_15, mean_error_LSTM_co2_full_15 ],
              [MSE_LSTM_co2_noise_full_15, R2_LSTM_co2_noise_full_15, accuracy_LSTM_co2_noise_full_15, mean_error_LSTM_co2_noise_full_15]],
              columns=['MSE', 'R2', 'accuracy', 'mean_error'], index=['Room 4 CO_2', 'Room 4 CO_2 + Sound', 'Full CO_2', 'Full CO_2 + Sound'])


# In[ ]:




