import time
from selenium import webdriver
from selenium.webdriver.support.select import Select
# %%

import json
import requests
import urllib
import pandas as pd
from pandas.io.json import json_normalize

# response = requests.get("https://edgetech.avuity.com/VuSpace/api/real-time-occupancy/get-by-area?access-token"
#                         f"=Futo24i1PcUZ_HnZ&startTs={start}&endTs={end}")
#
# # print(response.status_code)
#
# real_time_json = response.json()
# real_time_str = json.dumps(real_time_json)
# real_time_occ_dict = json.loads(real_time_str)
# real_time_occ_data_normalised = json_normalize(real_time_occ_dict['items'])
# real_time_occ_full = pd.DataFrame.from_dict(real_time_occ_data_normalised)
#
# real_time_occ_full = real_time_occ_full.dropna()
#
# real_time_occ = real_time_occ_full[['areaName', 'lastLogTs', 'dBs', 'occupancy', 'capacity']]

# %%
login_url = 'https://edgetech.avuity.com/VuSpace/site/login'

driver = webdriver.Chrome()
driver.implicitly_wait(5)
driver.get(login_url)
#
username = driver.find_element_by_id("loginform-username")
password = driver.find_element_by_id("loginform-password")
#
username.send_keys("mmikos")
password.send_keys("3dg3#Occupancy")

# driver.find_element_by_id("ts_and_cs").click()

# driver.find_element_by_css_selector('li.ts_and_cs')
# checkbox = driver.find_element_by_css_selector('label.checkbox')
# checkbox.click()

driver.find_element_by_name("login-button").click()

current_occupancy = 'https://edgetech.avuity.com/VuSpace/cur-occ/index'

real_time = driver.get(current_occupancy)

select_location = Select(driver.find_element_by_id('location'))
select_location.select_by_value("8")

driver.find_element_by_id('building')
driver.find_element_by_xpath('//*[@id="w0"]/div[1]/div[2]/div/span').click()
time.sleep(3)
driver.find_element_by_xpath('//*[@id="select2-building-container"]').click()
driver.find_element_by_id("select2-building-results").click()

driver.find_element_by_id('floor')
driver.find_element_by_xpath('//*[@id="w0"]/div[2]/div[1]/div/span').click()
time.sleep(3)
driver.find_element_by_xpath('//*[@id="select2-floor-container"]').click()
driver.find_element_by_id("select2-floor-results").click()

driver.find_element_by_id('areaType')
time.sleep(3)
driver.find_element_by_xpath('//*[@id="w0"]/div[2]/div[2]/div/span[2]/span[1]/span/ul/li/input').click()
driver.find_element_by_id('s2-togall-areaType').click()

#confirm
driver.find_element_by_xpath('//*[@id="w0"]/div[2]/div[4]/button').click()


print('It worked')