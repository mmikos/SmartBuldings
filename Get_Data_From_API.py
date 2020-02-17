import json
import pandas as pd
import requests
from pandas.io.json import json_normalize
from pandas.io.json import json_normalize
from datetime import timedelta
import urllib


class get_data_from_API:

    def __init__(self, start_date, end_date, freq, devices_list, sampling_rate):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.devices_list = devices_list
        self.sampling_rate = sampling_rate

    def get_avuity_data(self):

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        occupancy = pd.DataFrame()
        occupancy_selected_DF = pd.DataFrame(columns=['occupancy'])

        for day in range(int((end - start).days / self.freq)):
            date = start
            next_date = date + timedelta(days=self.freq)

            start_url = urllib.parse.quote(str(date))
            next_url = urllib.parse.quote(str(next_date))

            response = requests.get(
                "https://edgetech.avuity.com/VuSpace/api/report-occupancy-by-area/index?access-token"
                f"=Futo24i1PcUZ_HnZ&startTs={start_url}&endTs={next_url}")

            occupancy_json = response.json()
            occupancy_str = json.dumps(occupancy_json)
            occupancy_data_dict = json.loads(occupancy_str)
            occupancy_data_normalised = json_normalize(occupancy_data_dict['items'])
            occupancy_to_DF = pd.DataFrame.from_dict(occupancy_data_normalised)

            occupancy = occupancy.append(occupancy_to_DF, ignore_index=False)
            start = date + timedelta(days=self.freq)

        for room in self.devices_list.iloc[2, :]:
            occupancy_selected = occupancy.loc[occupancy['areaName'] == f'{room}', ['startTs', 'areaName', 'occupancy']]
            date_occupancy_selected = occupancy_selected.set_index('startTs')
            date_occupancy_selected.index = pd.to_datetime(date_occupancy_selected.index, utc=True)
            date_occupancy_agg = date_occupancy_selected.resample(self.sampling_rate).median().ffill().astype(int)
            occupancy_selected_DF = occupancy_selected_DF.append(date_occupancy_agg.reset_index(), ignore_index=False)

        occupancy_selected = occupancy_selected_DF.sort_values(by='startTs')
        occupancy = occupancy_selected.set_index('startTs')

        return occupancy

    def get_awair_data(self):

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        awair = pd.DataFrame()

        for day in range(int((end - start).days / self.freq)):
            date = start
            next_date = date + timedelta(days=self.freq)

            start_iso = date.isoformat()
            end_iso = next_date.isoformat()

            headers = {
                'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNTE4MDgifQ.HnZ_258AsEfbYLzmpK_g4jbTItIYbEQh_UaxCDO0S88',
            }

            params = (
                ('from', start_iso),
                ('to', end_iso),
            )

            awair_dataset = pd.DataFrame(columns=['comp', 'value', 'timestamp', 'score'])

            for device_id, room in zip(self.devices_list.iloc[1, :], self.devices_list.iloc[2, :]):
                response_awair = requests.get(
                    f'http://developer-apis.awair.is/v1/orgs/1097/devices/awair-omni/{device_id}/air-data'
                    f'/15-min-avg', headers=headers, params=params)
                awair_json = response_awair.json()
                awair_str = json.dumps(awair_json)
                awair_data_dict = json.loads(awair_str)
                awair_norm = pd.io.json.json_normalize(awair_data_dict["data"], record_path="sensors",
                                                       meta=['timestamp', 'score'])
                awair_sensor_code = [room for _ in range(len(awair_norm))]
                awair_norm['sensor_name'] = awair_sensor_code
                awair_dataset = awair_dataset.append(awair_norm, ignore_index=True)

            awair = awair.append(awair_dataset, ignore_index=False)
            start = date + timedelta(days=self.freq)

        awair = awair.sort_values(by='timestamp', ascending=True)
        awair = awair.set_index('timestamp')
        awair.index = pd.to_datetime(awair.index, utc=True)

        co2 = awair.loc[awair['comp'] == 'co2', ['value']]
        co2 = co2.resample(self.sampling_rate).mean().ffill()

        noise = awair.loc[awair['comp'] == 'spl_a', ['value']]
        noise = noise.resample(self.sampling_rate).mean().ffill()

        humidity = awair.loc[awair['comp'] == 'humid', ['value']]
        humidity = humidity.resample(self.sampling_rate).mean().ffill()

        temperature = awair.loc[awair['comp'] == 'temp', ['value']]
        temperature = temperature.resample(self.sampling_rate).mean().ffill()

        light = awair.loc[awair['comp'] == 'lux', ['value']]
        light = light.resample(self.sampling_rate).mean().ffill()

        return co2, noise, humidity, temperature, light
