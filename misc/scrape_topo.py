from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import time
import sys
import os.path
import os

out_file = 'topo_all1.csv'
# output file
header = [['material_id', 'p1', 'topo_class', 'p2', 'sub_class', 'p3', 'cross_type']]
df = pd.DataFrame(header)
df.to_csv(out_file, sep=';', header=None, index=False, mode='w')

data = pd.read_csv('./fetch_MPdata.csv', sep=';', header=0, index_col=None)
data = data.iloc[:6000]
ids = data['material_id']

# navigate to target website
options = webdriver.ChromeOptions()
options.add_argument("headless") # comment this for visualization
driver_path = os.getcwd() + "/chromedriver-linux" # google-chrome driver for Linux systems
#driver_path = os.getcwd() + "/chromedriver-mac" # google-chrome driver for macOS
driver = webdriver.Chrome(executable_path=driver_path,options=options)

data_out = []
cnt = 0
for material_id in ids:
    cnt += 1
    if cnt % 50 == 0:
        print('processed {} materials'.format(cnt))
        data_out = pd.DataFrame(data_out)
        data_out.to_csv(out_file, sep=';', header=None, index=False, mode='a')
        data_out = []

    try:
        idata = [material_id]
        driver.get("https://materialsproject.org/materials/"+material_id)
        time.sleep(5)
        elec = driver.find_element_by_id("electronic-structure")
        time.sleep(1)
        topo = elec.find_element_by_id("topological-data")
        time.sleep(1)
        table = topo.find_element_by_css_selector('tr')
        idata += table.text.split('\n')
        data_out.append(idata)
    except:
        continue

driver.quit()

