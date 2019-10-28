import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select


def open_export_file_window():
    export = driver.find_element(By.XPATH, '//*[@id="export-chart"]/a')
    time.sleep(5)
    export.click()

    element = driver.find_element_by_xpath('//*[@id="options"]/div[3]')
    attributeValue = element.get_attribute("style")

    if (attributeValue == "display: block;") == True:
        driver.find_element(By.XPATH, '//*[@id="options"]/div[3]')
        csv = driver.find_element(By.XPATH, '//*[@id="format_xls"]')
        csv.click()
        driver.find_element(By.XPATH, '//*[@id="options"]/div[5]/input').click()
    else:
        pass


login_url = 'https://leesmanindex.co.uk/login'

driver = webdriver.Chrome()
driver.implicitly_wait(30)
driver.get(login_url)
#
username = driver.find_element_by_id("person_session_email")
password = driver.find_element_by_id("person_session_password")
#
username.send_keys("fv@edge.tech")
password.send_keys("EDGETechnologies!")

# driver.find_element_by_id("ts_and_cs").click()

driver.find_element_by_css_selector('li.ts_and_cs')
checkbox = driver.find_element_by_css_selector('label.checkbox')
checkbox.click()

driver.find_element_by_name("commit").click()

results_url = 'https://leesmanindex.co.uk/admin/results'

survey_results = driver.get(results_url)

EV_Box_Ams = 'https://leesmanindex.co.uk/admin/results/EV-BoxAMS2019/analytics?scoped_question_id=summary&survey_type=office'
Ebbinge = 'https://leesmanindex.co.uk/admin/results/Ebbinge2019/analytics?scoped_question_id=summary&survey_type=office'
Epicenter_Ams = 'https://leesmanindex.co.uk/admin/results/EpicenterAMS2019/analytics?scoped_question_id=summary&survey_type=office'
SIG_Ams = 'https://leesmanindex.co.uk/admin/results/SIGAMS2019/analytics?scoped_question_id=summary&survey_type=office'
DFFRNT = 'https://leesmanindex.co.uk/admin/results/DFFRNTMedia2019/analytics?scoped_question_id=summary&survey_type=office'
EDGE_Ams_US_Berlin = 'https://leesmanindex.co.uk/admin/results/Edgenlnyger2019/analytics?scoped_question_id=summary&survey_type=office'

building_list = [EV_Box_Ams, Ebbinge, Epicenter_Ams, SIG_Ams, DFFRNT, EDGE_Ams_US_Berlin]
survey_questions = []

survey_questions = ['summary', '10', '11', '14', '15', '16', '17', 'MOB', '1857', '1856', '8', '9', 'EMO', 'OFFIMP',
                    'report-page-26',
                    'report-page-27', 'report-page-28', 'report-page-15', 'report-page-16', 'report-page-17',
                    'report-page-2',
                    'report-page-4', 'report-page-5', 'report-page-3', 'report-page-1', 'report-page-18',
                    'report-page-19', 'report-page-20', 'report-page-6', 'report-page-7', 'report-page-8',
                    'report-page-21', 'report-page-22', 'report-page-23', 'report-page-9', 'report-page-11',
                    'report-page-10', 'report-page-24', 'report-page-25',
                    'OVGEDGEFREQ', 'OVGEDGESAT', 'OVGEDGEIMP', 'OVGEDGEIMP2', 'OVGEDGEIMP3', 'OVGEDGEIMP4']

#
# building = building_list[5]
# question = '12


for building in building_list:
    driver.get(building)

    for question in survey_questions:
        select = Select(driver.find_element_by_id('scoped_question_id'))
        select.select_by_value(question)

        exists_table = len(driver.find_elements(By.XPATH, '//*[@id="show-table-link"]'))
        exists_report = len(driver.find_elements(By.XPATH, '//*[@id="show-report-link"]'))

        elements = [exists_report, exists_table]

        if all(elements) > 0:

            attributeValueReport = driver.find_element(By.XPATH, '//*[@id="show-report-link"]').get_attribute("class")
            attributeValueTable = driver.find_element(By.XPATH, '//*[@id="show-table-link"]').get_attribute("class")

            if attributeValueReport == 'active':
                open_export_file_window()

            elif attributeValueTable == 'active':
                open_export_file_window()

        elif any(elements) > 0:

            if exists_report > 0:
                driver.find_element(By.XPATH, '//*[@id="show-report-link"]').click()
                open_export_file_window()

            if exists_table > 0:
                driver.find_element(By.XPATH, '//*[@id="show-table-link"]').click()
                open_export_file_window()

        elif all(elements) == 0:
            open_export_file_window()
        else:
            pass

