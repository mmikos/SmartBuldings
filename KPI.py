import pandas as pd

def caclulate_KPIs(measurement: str, data_pivot):

    if measurement == 'CO2':

        CO2_KPI = pd.DataFrame
        data_pivot['CO2_KPI'] = CO2_KPI

        data_pivot_filtered = data_pivot['CO2'] < 400
        data_pivot_CO2 = data_pivot[data_pivot_filtered]
        data_pivot_CO2['CO2_KPI'] = 0

        data_pivot_filtered = data_pivot['CO2'] >= 400
        data_pivot_CO2_cond = data_pivot[data_pivot_filtered]
        data_pivot_CO2_cond['CO2_KPI'] = (1 - (- 0.05 * data_pivot_CO2_cond['CO2'] + 120) / 100)
        data_pivot_CO2 = data_pivot_CO2.append(data_pivot_CO2_cond, ignore_index=False, verify_integrity=False, sort=None)


        data_pivot_filtered = ((data_pivot['CO2'] < 0) & (data_pivot['CO2'] > 2500))
        data_pivot_CO2_cond = data_pivot[data_pivot_filtered]
        data_pivot_CO2_cond['CO2_KPI'] = 'sensor malfunction'
        data_pivot_CO2 = data_pivot_CO2.append(data_pivot_CO2_cond, ignore_index=False, verify_integrity=False, sort=None)


        data_pivot_CO2 = data_pivot_CO2.sort_values('CO2_KPI')

        return data_pivot_CO2

    if measurement == 'Temperature':
        Temperature_KPI = pd.DataFrame
        data_pivot['Temperature_KPI'] = Temperature_KPI

        data_pivot_filtered = (data_pivot['Temperature'] >= 16 & data_pivot['Temperature'] < 18)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.055

        data_pivot_filtered = (data_pivot['Temperature'] >= 18 & data_pivot['Temperature'] < 20)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.05

        data_pivot_filtered = (data_pivot['Temperature'] >= 20 & data_pivot['Temperature'] <= 24)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.04

        data_pivot_filtered = (data_pivot['Temperature'] > 24 & data_pivot['Temperature'] <= 26)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.045

        data_pivot_filtered = (data_pivot['Temperature'] > 26 & data_pivot['Temperature'] <= 28)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.055

        data_pivot_filtered = (data_pivot['Temperature'] > 28 & data_pivot['Temperature'] <= 30)
        data_pivot_Temperature = data_pivot[data_pivot_filtered]
        data_pivot_Temperature['Temperature_KPI'] = -0.065

    if measurement == 'Humidity':
        Humidity_KPI = pd.DataFrame
        data_pivot['Humidity_KPI'] = Humidity_KPI

        data_pivot_filtered = ((2.5863 * data_pivot['Humidity'] - 28.896) > 100)
        data_pivot_Humidity = data_pivot[data_pivot_filtered]
        data_pivot_Humidity['Humidity_KPI'] = 0

        data_pivot_filtered = ((2.5863 * data_pivot['Humidity'] - 28.896) < 100)
        data_pivot_Humidity = data_pivot[data_pivot_filtered]
        data_pivot_Humidity['Humidity_KPI'] = (- (2.5863 * data_pivot_Humidity['Humidity'] - 28.896) / 100)

    if measurement == 'Light':
        Light_KPI = pd.DataFrame
        data_pivot['Light_KPI'] = Light_KPI

        data_pivot_filtered = (data_pivot['Light'] > 500)
        data_pivot_Light = data_pivot[data_pivot_filtered]
        data_pivot_Light['Light_KPI'] = - 0.025

        data_pivot_filtered = (data_pivot['Light'] == 500)
        data_pivot_Light = data_pivot[data_pivot_filtered]
        data_pivot_Light['Light_KPI'] = - 0.01

        data_pivot_filtered = (data_pivot['Light'] < 500)
        data_pivot_Light = data_pivot[data_pivot_filtered]
        data_pivot_Light['Light_KPI'] = - 0.03

    if measurement == 'Noise':
        Noise_KPI = pd.DataFrame
        data_pivot['Noise_KPI'] = Noise_KPI

        data_pivot_filtered = (data_pivot['Noise'] >= 35 & data_pivot['Noise'] < 40)
        data_pivot_Noise = data_pivot[data_pivot_filtered]
        data_pivot_Noise['Noise_KPI'] = - 0.01

        data_pivot_filtered = (data_pivot['Noise'] >= 40 & data_pivot['Noise'] < 45)
        data_pivot_Noise = data_pivot[data_pivot_filtered]
        data_pivot_Noise['Noise_KPI'] = - 0.015

        data_pivot_filtered = (data_pivot['Noise'] >= 45)
        data_pivot_Noise = data_pivot[data_pivot_filtered]
        data_pivot_Noise['Noise_KPI'] = - 0.025

        data_pivot_filtered = (data_pivot['Noise'] < 35)
        data_pivot_Noise = data_pivot[data_pivot_filtered]
        data_pivot_Noise['Noise_KPI'] = 0

