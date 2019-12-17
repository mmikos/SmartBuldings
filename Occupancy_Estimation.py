import math
import matplotlib.pylab as pylab
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'small'}
pylab.rcParams.update(params)
import warnings

warnings.filterwarnings('ignore')


def fit_model(model, X_train, X_test, y_train, y_test):
    features_number = X_train.shape[1]

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    model.fit(X_train.reshape(-1, features_number), y_train.reshape(-1, 1))
    prediction = model.predict(X_test.reshape(-1, features_number))
    accuracy = model.score(X_test, y_test)
    MSE = math.sqrt(mean_squared_error(y_test, prediction))
    mean_error = np.mean(np.abs(y_test - prediction))

    confusion_matrix_calculated = confusion_matrix(y_test, prediction)

    return prediction, accuracy, MSE, mean_error, confusion_matrix_calculated


def plot_confusion_matrix(confusion_matrix_calculated,
                          accuracy,
                          model_name):
    matrix_percentage = confusion_matrix_calculated.astype(dtype=np.float32)

    rows = confusion_matrix_calculated.shape[0]
    columns = confusion_matrix_calculated.shape[1]

    for column in range(columns):
        for row in range(rows):
            matrix_percentage[row, column] = matrix_percentage[row, column] / \
                                             np.sum(confusion_matrix_calculated, axis=1)[row]

    plt.figure(figsize=(9, 9))
    sns.heatmap(matrix_percentage, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = f'Accuracy Score for {model_name}: {np.round(accuracy, 4) * 100} %'
    plt.title(all_sample_title, size=15)
    plt.show()


def filter_measurements_data(measure, name, business_hours, weekends):
    if business_hours:
        if name == 'co2':
            measure = measure.shift(0)
        measure = measure.between_time('8:00', '18:00')

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


def resample_occupancy(occupancy_binary, sampling_rate, devices_list):
    occupancy_selected_DF = pd.DataFrame(columns=['Value'])

    for room in devices_list.iloc[3, :]:
        occupancy_selected = occupancy_binary.loc[occupancy_binary['SpaceName'] == f'{room}', ['Datetime', 'Value']]
        date_occupancy_selected = occupancy_selected.set_index('Datetime')
        date_occupancy_selected.index = pd.to_datetime(date_occupancy_selected.index, utc=True)
        date_occupancy_agg = date_occupancy_selected.resample(sampling_rate).min().ffill().astype(int)
        occupancy_selected_DF = occupancy_selected_DF.append(date_occupancy_agg.reset_index(), ignore_index=False)

    occupancy_selected = occupancy_selected_DF.sort_values(by='Datetime')
    occupancy = occupancy_selected.set_index('Datetime')

    occupancy = occupancy.rename(columns={'Value': 'occupancy'})

    occupancy = occupancy.where(occupancy == 0, 1)

    return occupancy


def plot_ROC_curve(X_test, y_test, model, model_name: str):
    y_test = y_test.astype('int')

    # generate a no skill prediction (majority class)
    random_class_probability = [0 for _ in range(len(y_test))]

    predicted_class_probability = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    predicted_class_probability = predicted_class_probability[:, 1]

    random_class_auc = roc_auc_score(y_test, random_class_probability)
    predicted_class_auc = roc_auc_score(y_test, predicted_class_probability)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (random_class_auc))
    print('Logistic: ROC AUC=%.3f' % (predicted_class_auc))

    # calculate roc curves
    random_class_fpr, random_class_tpr, _ = roc_curve(y_test, random_class_probability)
    predicted_class_fpr, predicted_class_tpr, _ = roc_curve(y_test, predicted_class_probability)
    # plot the roc curve for the model
    pyplot.plot(random_class_fpr, random_class_tpr, linestyle='--', label='Random')
    pyplot.plot(predicted_class_fpr, predicted_class_tpr, marker='.', label='Predicted')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {model_name}', size=15)
    pyplot.legend()
    pyplot.show()


start_date = "2019-11-18 08:00"
end_date = "2019-12-18 08:00"
freq = 5
sampling_rate = '15T'
devices_list_full = pd.DataFrame([['OLY-A-413', 'OLY-A-414', 'OLY-A-415', 'OLY-A-416', 'OLY-A-417'],
                                  ['9033', '9049', '8989', '7675', '7663'],
                                  ['ROOM 1', 'ROOM 2', 'ROOM 3', 'ROOM 4', 'ROOM 5'],
                                  ['Room 4.1', 'Room 4.2', 'Room 4.3', 'Room 4.4', 'Room 4.5']])
devices_list_room_1 = pd.DataFrame([['OLY-A-413'],
                                    ['9033'],
                                    ['ROOM 1'],
                                    ['Room 4.1']])
devices_list_room_2 = pd.DataFrame([['OLY-A-414'],
                                    ['9049'],
                                    ['ROOM 2'],
                                    ['Room 4.2']])
devices_list_room_3 = pd.DataFrame([['OLY-A-415'],
                                    ['8989'],
                                    ['ROOM 3'],
                                    ['Room 4.3']])
devices_list_room_4 = pd.DataFrame([['OLY-A-416'],
                                    ['7675'],
                                    ['ROOM 4'],
                                    ['Room 4.4']])
devices_list_room_5 = pd.DataFrame([['OLY-A-417'],
                                    ['7663'],
                                    ['ROOM 5'],
                                    ['Room 4.5']])

devices_list = devices_list_room_4

occupancy_binary_part1 = pd.read_csv('data_sets/binary_occupancy_Nov.csv')
occupancy_binary_part2 = pd.read_csv('data_sets/binary_occupancy_Dec1.csv')
occupancy_binary_part3 = pd.read_csv('data_sets/binary_occupancy_Dec2.csv')

occupancy_binary = occupancy_binary_part1.append([occupancy_binary_part2, occupancy_binary_part3])
occupancy = resample_occupancy(occupancy_binary, sampling_rate, devices_list)

load = get_data_from_API(start_date, end_date, freq, devices_list, sampling_rate)

# occupancy = load.get_avuity_data()

co2, noise, humidity, temperature = load.get_awair_data()

business_hours = True
weekends = True

co2 = filter_measurements_data(co2, 'co2', business_hours, weekends)
noise = filter_measurements_data(noise, 'noise', business_hours, weekends)
humidity = filter_measurements_data(humidity, 'humidity', business_hours, weekends)
temperature = filter_measurements_data(temperature, 'temperature', business_hours, weekends)

data = create_dataset(occupancy, co2, noise, humidity)

plt.hist(data['occupancy'], 2, histtype='bar')
plt.show()

data_cut = (data.loc[(data.index.day == 25) & (data.index.month == 11)]).append(data.loc[(data.index.day == 26) &
                                                                                         (data.index.month == 11)])



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
#
# ax2.set_ylabel('Noise', color=color)
#
# ax2.plot(data_cut['noise'], color=color)
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(f'Occupancy vs CO_2')

scaler = preprocessing.StandardScaler()
data = data.to_numpy()
no_features = data.shape[1] - 1
data[:, 1:no_features + 1] = scaler.fit_transform(data[:, 1:no_features + 1])

# data[:, 0][np.where(data[:, 0] > 0)] = 1

Results_MLP_list = []
Results_LSTM_list = []
Results_CNN_list = []
Results_CNN_LSTM_list = []

number_of_epochs = 1000
window_size = 32
batch_size = 64

max_depth = 6
n_estimators = 150

R2_decision_tree, MSE_decision_tree, y_predicted, y_test_dt = Decision_Tree_Regression(
    data[:, 1:no_features + 1].reshape(-1, no_features),
    data[:, 0].reshape(-1, 1),
    max_depth,
    n_estimators, 'plot', 'measure', 'occ')

mean_error_DT = np.mean(np.abs(y_test_dt - y_predicted))

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(data[:, 1:no_features + 1], data[:, 0],
                                                                            test_size=1 / 3, random_state=42,
                                                                            shuffle=True)

model_Random_Forest = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=0, class_weight='balanced')
# model_Random_Forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, class_weight='balanced')
prediction_Random_Forest, accuracy_Random_Forest, MSE_Random_Forest, mean_error_Random_Forest, cm_Random_Forest = \
    fit_model(model_Random_Forest, X_train_class, X_test_class, y_train_class, y_test_class)

plot_confusion_matrix(cm_Random_Forest, accuracy_Random_Forest, 'Random Forest')
plot_ROC_curve(X_test_class, y_test_class, model_Random_Forest, 'Random Forest')

model_SVC = SVC(C=200, kernel='rbf', class_weight='balanced', gamma='auto', probability=True)
# model_SVC = SVC(C=100, kernel='rbf', class_weight='balanced', gamma='auto')
prediction_SVC, accuracy_SVC, MSE_SVC, mean_error_SVC, cm_SVC = \
    fit_model(model_SVC, X_train_class, X_test_class, y_train_class, y_test_class)

plot_confusion_matrix(cm_SVC, accuracy_SVC, 'SVC')
plot_ROC_curve(X_test_class, y_test_class, model_SVC, 'SVC')

log_regres = LogisticRegression(C=100, random_state=0, class_weight='balanced')

prediction_log_regres, accuracy_log_regres, MSE_log_regres, mean_error_log_regres, cm_log_regres = \
    fit_model(log_regres, X_train_class, X_test_class, y_train_class, y_test_class)

plot_confusion_matrix(cm_log_regres, accuracy_log_regres, 'Logistic Regression')
plot_ROC_curve(X_test_class, y_test_class, log_regres, 'Logistic Regression')

batches, labels = prepare_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(batches, labels, test_size=1 / 3, random_state=42)

metrics = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]

regularization_penalty = 0.000001

accuracy_ANN, prediction_ANN, cm_ANN, model_ANN = ANN_classify(X_train, X_test, y_train, y_test, 100, number_of_epochs,
                                                    window_size, metrics, regularization_penalty)

metrics_results = pd.DataFrame([accuracy_ANN], columns=['loss', 'TP', 'FP', 'TN', 'FN', 'accuracy', 'precision',
                                                        'recall', 'AUC'])

plot_confusion_matrix(cm_ANN, metrics_results.iloc[0, 6], 'Neural Network')
ANN_summary = classification_report(y_test, prediction_ANN)
plot_ROC_curve(X_test, y_test, model_ANN, 'Neural Network')

number_of_nodes_MLP = 64
model_MLP = MLP(window_size, no_features, number_of_nodes_MLP)
training_history_MLP = model_MLP.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_MLP = model_MLP.predict(X_test)
MSE_MLP, R2_MLP, residuals_MLP, accuracy_MLP, mean_error_MLP = model_MLP.evaluate(prediction_MLP, X_test, y_test)

# # Results_MLP_list.append([R2_MLP, MSE_MLP, window_size, batch_size, number_of_nodes])

number_of_nodes_LSTM = 32
model_LSTM = LSTM(window_size, no_features, number_of_nodes_LSTM)
training_history_LSTM = model_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_LSTM = model_LSTM.predict(X_test)
MSE_LSTM, R2_LSTM, residuals_LSTM, accuracy_LSTM, mean_error_LSTM = model_LSTM.evaluate(prediction_LSTM, X_test, y_test)

# # Results_LSTM_list.append([MSE_LSTM, R2_LSTM, window_size, batch_size, number_of_nodes])

number_of_filters_CNN = 64
model_CNN = CNN(window_size, no_features, number_of_filters_CNN)
training_history_CNN = model_CNN.fit(X_train, X_test, y_train, y_test, number_of_epochs, batch_size)
prediction_CNN = model_CNN.predict(X_test)
MSE_CNN, R2_CNN, residuals_CNN, accuracy_CNN, mean_error_CNN = model_CNN.evaluate(prediction_CNN, X_test, y_test)

# Results_CNN_list.append([MSE_CNN, R2_CNN, window_size, batch_size, number_of_nodes])

number_of_filters_CNN_LSTM = 64
number_of_nodes_CNN_LSTM = 64
model_CNN_LSTM = CNN_LSTM(window_size, no_features, number_of_filters_CNN_LSTM, number_of_nodes_CNN_LSTM)
training_history_CNN_LSTM = model_CNN_LSTM.fit(X_train, X_test, y_train, y_test, number_of_epochs)
prediction_CNN_LSTM = model_CNN_LSTM.predict(X_test)
MSE_CNN_LSTM, R2_CNN_LSTM, residuals_CNN_LSTM, accuracy_CNN_LSTM, mean_error_CNN_LSTM = model_CNN_LSTM.evaluate(
    prediction_CNN_LSTM, X_test, y_test)

# Results_CNN_LSTM_list.append([MSE_CNN_LSTM, MSE_MLP, window_size, batch_size, number_of_nodes_CNN_LSTM, number_of_filters_CNN_LSTM])

# Results_MLP = pd.DataFrame(Results_MLP_list, columns=['R2', 'MSE', 'window_size', 'batch_size', 'number_of_nodes_MLP'])
# Results_MLP = Results_MLP.sort_values(['R2'], ascending=False)


model_CNN_LSTM.plot_results(y_test, residuals_CNN_LSTM, prediction_CNN_LSTM)

plt.figure(figsize=(12, 10))
plt.plot(training_history_LSTM.history['loss'])
plt.plot(training_history_LSTM.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.figure(figsize=(12, 10))
plt.plot(y_test, label='Real values', color='green')
plt.plot(prediction_LSTM, label='Predicted values', color='red')
plt.title('Real vs predicted using LSTM')
plt.legend()

plt.figure(figsize=(12, 10))
plt.plot(y_test, label='Real values', color='green')
plt.plot(np.round(prediction_MLP, 0), label='Predicted values', color='red')
plt.title('Real vs predicted using MLP')
plt.legend()

plt.figure(figsize=(12, 10))
plt.plot(y_test, label='Real values', color='green')
plt.plot(np.round(prediction_CNN_LSTM, 0), label='Predicted values', color='red')
plt.title('Real vs predicted using CNN_LSTM')
plt.legend()

plt.figure(figsize=(12, 10))
plt.plot(y_test, label='Real values', color='green')
plt.plot(np.round(prediction_CNN, 0), label='Predicted values', color='red')
plt.title('Real vs predicted using CNN')
plt.legend()

plt.show()

print('whatever')

# R2_ANN, MSE_ANN, y_predicted, training_history = ANN_regress(data, number_of_nodes, dropout_rate, window_size,
#                                                              batch_size, number_of_epochs, regularization_penalty)

# for nodes in tqdm(number_of_nodes):
#
#         R2_ANN, MSE_ANN, y_predicted, training_history = ANN_regress(data, nodes, dropout_rate, window_size,
#                                                              batch_size, number_of_epochs, regularization_penalty)
#         Results_ANN_list.append([R2_ANN, MSE_ANN, nodes])
#
# Results_ANN = pd.DataFrame(Results_ANN_list, columns=['R2', 'MSE', 'nodes'])
# Results_ANN = Results_ANN.sort_values(['R2'], ascending=False)


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

# scores_test, pred_test, confusion_matrix = ANN_classify(X_train, X_test, y_train, y_test, number_of_nodes,
#                                                         number_of_layers, dropout_rate, number_of_epochs,
#                                                         regularization_penalty)


# degree = 3
# # Perform polynomial transformation
# polynomial_features = PolynomialFeatures(degree=degree)
# sensor_values_polynomial = polynomial_features.fit_transform(X_train)
#
# # Fit the model
# model = sm.OLS(y_train, sensor_values_polynomial).fit()
# # fit_regularized(alpha=0.2, L1_wt=0.5)
# test0 = np.array([[1], [22], [60]]).T
# score_values_predicted = model.predict(polynomial_features.fit_transform(X_test))
#
# rsquared = model.rsquared_adj
# MSE = math.sqrt(mean_squared_error(y_test, score_values_predicted))
#
# print(model.summary())
# X_test_sorted, y_predicted_sorted = sort_for_plotting(X_test, score_values_predicted)
# # Plot original data
# plt.scatter(X_test, y_test, color='red')
# # plt.scatter(X[:, 1], y, color='blue')
#
# plt.plot(X_test_sorted, y_predicted_sorted,
#          label=f"Degree {1}," + f" $R^2$: {round(rsquared, 3)}, MSE: {round(MSE, 3)}")
#
# plt.legend(loc='upper right')
# plt.xlabel("CO_2")
# plt.ylabel("Occupancy")
# plt.title(f'Satisfaction vs indoor conditions (Polynomial Regression)')
# plt.show()

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
# outlier_detection = KMeans(n_clusters=6, random_state=0)

# sns.boxplot(x= occupancy_selected.reset_index(drop=True).iloc[:, 0])
# plt.show()

# z = np.abs(stats.zscore(reg_data_no_na[['co2']]))
# print(z)
# threshold = 3
# print(np.where(z > 3))

# reg_data_no_na = reg_data_no_na[(z < 3).all(axis=1)]
# clusters = outlier_detection.fit_predict(co2[['value']])


# co2_date = co2_date.between_time('9:15', '16:45')
# noise_date = noise_date.between_time('9:15', '16:45')
# occupancy_selected_date = occupancy_selected_date.between_time('9:15', '16:45')
