import math

def HI_calculation(sensor_temperature, sensor_humidity):
    """
    :param sensor_temperature: array of temperature readings
    :param sensor_humidity: array of humidity readings
    :param number_of_samples: number of generated measurements
    :return: Heat Index
    """
    heat_index = []
    a: float = 17.27
    b: float = 237.3

    alfa = ((a * sensor_temperature) / (b + sensor_temperature)) + math.log(
        sensor_humidity / 100)

    D = (b * alfa) / (a - alfa)

    HI = sensor_temperature - 1.0799 * math.exp(0.03755 * sensor_temperature) * (
            1 - math.exp(0.0801 * (D - 14)))

    heat_index.append(HI)

    return heat_index

sensor_temperature = 21.48
sensor_humidity = 30.3


heat_index = HI_calculation(sensor_temperature, sensor_humidity)

print("x")