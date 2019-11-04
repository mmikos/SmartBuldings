import numpy as np
import pandas as pd
import math

import pywt


def HI_calculation(sensor_temperature, sensor_humidity, number_of_samples):
    """
    :param sensor_temperature: array of temperature readings
    :param sensor_humidity: array of humidity readings
    :param number_of_samples: number of generated measurements
    :return: Heat Index
    """
    heat_index = []
    a: float = 17.27
    b: float = 237.3

    for cell in range(number_of_samples):
        alfa = ((a * sensor_temperature[cell]) / (b + sensor_temperature[cell])) + math.log(
            sensor_humidity[cell] / 100)

        D = (b * alfa) / (a - alfa)

        HI = sensor_temperature[cell] - 1.0799 * math.exp(0.03755 * sensor_temperature[cell]) * (
                1 - math.exp(0.0801 * (D - 14)))

        heat_index.append(HI)

    return heat_index


def generate_satisfaction_scores(generated_data, number_of_samples):
    """
    :param generated_data: array of generated sensor measurements
    :param number_of_samples: number of generated measurements
    :return: array with sensors and generated satisfaction scores
    """

    for cell in range(number_of_samples):

        ### Temperature
        if (generated_data[cell, 0] > 17) and (generated_data[cell, 0] <= 29):
            generated_data[cell, 6] = np.random.uniform(0.2, 0.4)

        if (generated_data[cell, 0] > 19) and (generated_data[cell, 0] <= 27):
            generated_data[cell, 6] = np.random.uniform(0.4, 0.6)

        if (generated_data[cell, 0] > 21) and (generated_data[cell, 0] <= 25):
            generated_data[cell, 6] = np.random.uniform(0.6, 0.8)

        if (generated_data[cell, 0] > 22) and (generated_data[cell, 0] <= 24):
            generated_data[cell, 6] = np.random.uniform(0.8, 1)

        ### Humidity
        if (generated_data[cell, 1] > 15) and (generated_data[cell, 1] <= 85):
            generated_data[cell, 7] = np.random.uniform(0.2, 0.4)

        if (generated_data[cell, 1] > 20) and (generated_data[cell, 1] <= 80):
            generated_data[cell, 7] = np.random.uniform(0.4, 0.6)

        if (generated_data[cell, 1] > 25) and (generated_data[cell, 1] <= 75):
            generated_data[cell, 7] = np.random.uniform(0.6, 0.8)

        if (generated_data[cell, 1] > 30) and (generated_data[cell, 1] <= 70):
            generated_data[cell, 7] = np.random.uniform(0.8, 1)

        ### Apparent temperature - Heat Index (Temperature accounted for humidity)
        if (generated_data[cell, 2] > 17) and (generated_data[cell, 2] <= 29):
            generated_data[cell, 8] = np.random.uniform(0.2, 0.4)

        if (generated_data[cell, 2] > 19) and (generated_data[cell, 2] <= 27):
            generated_data[cell, 8] = np.random.uniform(0.4, 0.6)

        if (generated_data[cell, 2] > 21) and (generated_data[cell, 2] <= 25):
            generated_data[cell, 8] = np.random.uniform(0.6, 0.8)

        if (generated_data[cell, 2] > 22) and (generated_data[cell, 2] <= 24):
            generated_data[cell, 8] = np.random.uniform(0.8, 1)

        ### CO2
        if generated_data[cell, 3] <= 1800:
            generated_data[cell, 9] = np.random.uniform(0.1, 0.2)

        if (generated_data[cell, 3] > 1200) and (generated_data[cell, 3] <= 1500):
            generated_data[cell, 9] = np.random.uniform(0.2, 0.4)

        if (generated_data[cell, 3] > 1000) and (generated_data[cell, 3] <= 1200):
            generated_data[cell, 9] = np.random.uniform(0.4, 0.6)

        if (generated_data[cell, 3] > 800) and (generated_data[cell, 3] <= 1000):
            generated_data[cell, 9] = np.random.uniform(0.6, 0.8)

        if (generated_data[cell, 3] > 650) and (generated_data[cell, 3] <= 800):
            generated_data[cell, 9] = np.random.uniform(0.8, 0.9)

        if generated_data[cell, 3] <= 650:
            generated_data[cell, 9] = np.random.uniform(0.9, 1)

        ### Light
        if generated_data[cell, 4] >= 300:
            generated_data[cell, 10] = np.random.uniform(0.2, 0.4)

        if generated_data[cell, 4] >= 450:
            generated_data[cell, 10] = np.random.uniform(0.4, 0.6)

        if generated_data[cell, 4] >= 600:
            generated_data[cell, 10] = np.random.uniform(0.6, 0.8)

        if generated_data[cell, 4] >= 1000:
            generated_data[cell, 10] = np.random.uniform(0.8, 1)

        ### Noise
        if generated_data[cell, 5] <= 70:
            generated_data[cell, 11] = np.random.uniform(0.2, 0.4)

        if generated_data[cell, 5] <= 65:
            generated_data[cell, 11] = np.random.uniform(0.4, 0.6)

        if generated_data[cell, 5] <= 60:
            generated_data[cell, 11] = np.random.uniform(0.6, 0.8)

        if generated_data[cell, 5] <= 55:
            generated_data[cell, 11] = np.random.uniform(0.8, 1)

    return generated_data

def generate_null_model(generated_data, number_of_samples):
    """
    :param generated_data: array of generated sensor measurements
    :param number_of_samples: number of generated measurements
    :return: array with sensors and random satisfaction scores
    """

    for cell in range(number_of_samples):

    ### Temperature
        generated_data[cell, 6] = np.random.uniform(0.2, 1)
    ### Humidity
        generated_data[cell, 7] = np.random.uniform(0.2, 1)
    ### Heat Index
        generated_data[cell, 8] = np.random.uniform(0.2, 1)
    ### CO2
        generated_data[cell, 9] = np.random.uniform(0.2, 1)
    ### Light
        generated_data[cell, 10] = np.random.uniform(0.2, 1)
    ### Noise
        generated_data[cell, 11] = np.random.uniform(0.2, 1)

    return generated_data

def smoothing(y_values: np.ndarray, number_of_samples, wavelet):
    """
    Method that smooths a time series using wavelet transforms.

    ```
    :param y_values: y values of the original time series
    :param number_of_samples: desired length of the results
    :param wavelet: type of wavelet i.e. db2, db8 etc.
    :return: smoothed data series
    """

    levels = pywt.dwt_max_level(y_values.shape[0], wavelet)

    # Decompose getting only the details
    for _ in range(levels):
        y_values = pywt.downcoef(part='a', data=y_values, wavelet=wavelet)

    for _ in range(levels):
        details = np.zeros(y_values.shape)
        y_values = pywt.idwt(y_values, details, wavelet=wavelet)

    return y_values[:number_of_samples]


def generate_dataset_with_sensor_readings_and_satisfaction_scores(number_of_samples, noise_standard_deviation, wavelet):
    """
    :param number_of_samples: number of generated measurements
    :param noise_standard_deviation: standard deviation of a noise added to the generated sample
    :param wavelet: type of wavelet
    :return: generated dataset
    """
    # Generate a random number from an uniform distribution in a range given by KPIs

    sensor_temperature = [np.random.uniform(15, 32) for x in range(number_of_samples)]
    sensor_humidity = [np.random.uniform(20, 85) for x in range(number_of_samples)]
    sensor_heat_index = HI_calculation(sensor_temperature, sensor_humidity, number_of_samples)
    sensor_co2 = [np.random.uniform(250, 2000) for x in range(number_of_samples)]
    sensor_light = [np.random.uniform(100, 1500) for x in range(number_of_samples)]
    sensor_noise = [np.random.uniform(30, 75) for x in range(number_of_samples)]

    # Create an ndarray with all generated sensor measurements and fill the new columns of ndarray with the random
    # values between 15% and 20% representing guaranteed minimal satisfaction in each category

    generated_data = np.asarray(
        list(zip(sensor_temperature, sensor_humidity, sensor_heat_index, sensor_co2, sensor_light, sensor_noise,
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)],
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)],
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)],
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)],
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)],
                 [np.random.uniform(0.1, 0.2) for x in range(number_of_samples)])), dtype=np.float32)

    generated_data_with_scores = generate_satisfaction_scores(generated_data, number_of_samples)

    for i in range(6, 12):
        generated_data_with_scores = generated_data_with_scores[generated_data_with_scores[:, i - 6].argsort()]
        generated_data_with_scores[:, i] = smoothing(generated_data_with_scores[:, i], number_of_samples, wavelet)

    mean = 0
    noise = np.random.normal(mean, noise_standard_deviation, [number_of_samples, 6])

    generated_data_with_scores[:, 6:] = generated_data_with_scores[:, 6:] + noise

    # outisde bounds
    generated_data_with_scores[:, 6:][np.where(generated_data_with_scores[:, 6:] > 1)] = np.random.uniform(0.95, 1)
    generated_data_with_scores[:, 6:][np.where(generated_data_with_scores[:, 6:] < 0)] = np.random.uniform(0, 0.05)

    satisfaction_vs_sensors = pd.DataFrame(generated_data_with_scores,
                                           columns=['sensor_temperature', 'sensor_humidity', 'sensor_heat_index',
                                                    'sensor_air', 'sensor_light', 'sensor_noise',
                                                    'comfort_score_temperature', 'comfort_score_humidity',
                                                    'comfort_score_heat_index', 'comfort_score_air',
                                                    'comfort_score_light', 'comfort_score_noise'])

    generated_data_null = generate_null_model(generated_data, number_of_samples)

    satisfaction_vs_sensors_null = pd.DataFrame(generated_data_null,
                                                columns=['sensor_temperature', 'sensor_humidity', 'sensor_heat_index',
                                                         'sensor_air', 'sensor_light', 'sensor_noise',
                                                         'comfort_score_temperature', 'comfort_score_humidity',
                                                         'comfort_score_heat_index', 'comfort_score_air',
                                                         'comfort_score_light', 'comfort_score_noise'])

    return satisfaction_vs_sensors, satisfaction_vs_sensors_null