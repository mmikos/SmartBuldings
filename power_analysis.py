# Start Python Imports
import numpy as np
import pandas as pd
# Data Manipulation
from scipy.stats import ttest_ind
from tqdm import tqdm, tqdm_notebook, tnrange
import matplotlib.pyplot as plt
from satisfaction_score_data_generator import generate_dataset_with_sensor_readings_and_satisfaction_scores

# POWER ANALYSIS
confidence_level = 0.05
sample_size = 20
noise_standard_deviation = 0.05
measurement_name = 'air'
score_name = f'comfort_score_{measurement_name}'

satisfaction_vs_sensors, satisfaction_vs_sensors_null = generate_dataset_with_sensor_readings_and_satisfaction_scores(sample_size, noise_standard_deviation, 'db8')

score_values = satisfaction_vs_sensors[[score_name]]

score_values_null = satisfaction_vs_sensors_null[[score_name]]

t_statistics, p_value_t = ttest_ind(score_values, score_values_null)
print(f'T-statistics = {t_statistics}, p-value = {p_value_t}')


def effect_size(score_values, score_values_null):
    mean = np.mean(score_values)
    mean_null = np.mean(score_values_null)

    standard_deviation = np.std(np.column_stack((score_values, score_values_null)))

    cohen_d = (mean - mean_null) / standard_deviation

    return cohen_d


def test_hypothesis(sample_size, confidence_level, noise_standard_deviation, measurement_name):
    satisfaction_vs_sensors, satisfaction_vs_sensors_null = \
        generate_dataset_with_sensor_readings_and_satisfaction_scores(sample_size, noise_standard_deviation, 'db8')

    score_name = f'comfort_score_{measurement_name}'

    score_values = satisfaction_vs_sensors[[score_name]]
    score_values_null = satisfaction_vs_sensors_null[[score_name]]

    t_statistics, p_value_t = ttest_ind(score_values, score_values_null, equal_var = False)

    result = p_value_t < confidence_level

    return result


result = test_hypothesis(sample_size, confidence_level, noise_standard_deviation, measurement_name)
print(result)

sample_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
               250]
sample_size = [200, 250, 500]

alpha_levels = [0.05]
number_of_experiments = 100


def power_simulation_testing(number_of_experiments, sample_size, alpha_levels, noise_standard_deviation,
                             measurement_name):
    rejections = np.zeros(number_of_experiments, dtype=bool)
    test_power = []
    for size in tqdm(sample_size):
        for alpha in tqdm(alpha_levels):
            for experiment in tqdm(range(number_of_experiments)):
                rejections[experiment] = test_hypothesis(size, alpha, noise_standard_deviation, measurement_name)

            test_power.append([size, np.mean(rejections), alpha])

    return test_power


test_power_air = power_simulation_testing(number_of_experiments, sample_size, alpha_levels, noise_standard_deviation,
                                          'air')

test_power_temperature = power_simulation_testing(number_of_experiments, sample_size, alpha_levels,
                                                  noise_standard_deviation, 'heat_index')

test_power_noise = power_simulation_testing(number_of_experiments, sample_size, alpha_levels, noise_standard_deviation,
                                            'noise')

test_power_light = power_simulation_testing(number_of_experiments, sample_size, alpha_levels, noise_standard_deviation,
                                            'light')

test_power_table_air = pd.DataFrame(test_power_air, columns=['Sample Size', 'Power', 'Alpha'])
test_power_table_temperature = pd.DataFrame(test_power_temperature, columns=['Sample Size', 'Power', 'Alpha'])
test_power_table_noise = pd.DataFrame(test_power_noise, columns=['Sample Size', 'Power', 'Alpha'])
test_power_table_light = pd.DataFrame(test_power_light, columns=['Sample Size', 'Power', 'Alpha'])

scores = [test_power_table_air, test_power_table_temperature, test_power_table_noise, test_power_table_light]
names = ['Air', 'Temperature', 'Noise', 'Light']

i = 1

# Plot test power
plt.figure(figsize=(20, 18))
for plot in scores:

    plt.subplot(2, 2, i)

    for alpha in alpha_levels:
        plt.plot(plot.loc[plot['Alpha'] == alpha, 'Sample Size'],
                 plot.loc[plot['Alpha'] == alpha, 'Power'], 'o-',
                 label=f"Confidence level: {alpha}")

    plt.xlabel(f'Number of samples for {names[i - 1]}')
    plt.ylabel('Power')
    plt.axhline(y=0.8, color='r')
    plt.title('Test Power vs Sample Size')
    plt.legend()

    i = i + 1

plt.show()
