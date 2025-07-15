from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import os
from collections import defaultdict
import requests

# browserOptions = webdriver.ChromeOptions()
# browserOptions.add_argument('--proxy-server=ip:port)
# browser = webdriver.Chrome(chrome_options=browserOptions)

class Crawler_google_images:

    def __init__(self):
        self.url = None

    def init_browser(self, url):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument('--headless')
        browser = webdriver.Chrome(options=chrome_options)
        browser.get(url)
        browser.maximize_window()
        return browser

    def download_images(self, dataset, browser, keyword, round=2):
        picpath = '../data/'+ dataset + '/fig/' + keyword
        if not os.path.exists(picpath): os.makedirs(picpath)
        img_url_dic = []

        count = 0
        pos = 0
        for i in range(round):
            pos += 500
            js = 'var q=document.documentElement.scrollTop=' + str(pos)
            browser.execute_script(js)
            time.sleep(1)

            # img_elements = browser.find_elements_by_tag_name('img')
            img_elements = browser.find_elements(By.TAG_NAME, 'img')
            # browser.switch_to.frame(img_elements)
            for img_element in img_elements:
                wait = WebDriverWait(browser, 10)
                wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'img')))
                img_url = img_element.get_attribute('src')
                if isinstance(img_url, str):
                    if len(img_url) <= 200:
                        if 'images' in img_url:
                            if img_url not in img_url_dic:
                                try:
                                    img_url_dic.append(img_url)
                                    filename = picpath + "/" + str(count) + ".jpg"
                                    r = requests.get(img_url)
                                    with open(filename, 'wb') as f:
                                        f.write(r.content)
                                    f.close()
                                    count += 1
                                    print('this is '+str(count)+'st img')
                                    time.sleep(0.2)
                                    if count >= 10:
                                        break
                                except:
                                    print('failure')

    def run(self, dataset, keyword, url):
        self.__init__()
        browser = self.init_browser(url)

        if '/' in keyword:
            keyword_crowl = keyword.split()[0].replace(' ', '_')
        elif "\"" in keyword:
            keyword_crowl = keyword.split()[0].replace("\"", "")
        else:
            keyword_crowl = keyword.split()[0]

        self.download_images(dataset, browser, keyword_crowl, 1)
        browser.close()
        print(keyword + "\tDone")


if __name__ == '__main__':
    datasets = ["ICE14-IMG-TXT", "ICE0515-IMG-TXT", "ICE18-IMG-TXT", "GDELT-IMG-TXT"]
    def get_entity_time_ranges(dataset):
        time_ranges = defaultdict(dict)
        with open(f"../data/{dataset}/train.txt", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    s, _, o, t = parts[:4]
                    t = t.strip()
                    if s not in time_ranges:
                        time_ranges[s] = {'min_t': t, 'max_t': t}
                    if o not in time_ranges:
                        time_ranges[o] = {'min_t': t, 'max_t': t}
                    if t < time_ranges[s]['min_t']:
                        time_ranges[s]['min_t'] = t
                    if t > time_ranges[s]['max_t']:
                        time_ranges[s]['max_t'] = t
                    if t < time_ranges[o]['min_t']:
                        time_ranges[o]['min_t'] = t
                    if t > time_ranges[o]['max_t']:
                        time_ranges[o]['max_t'] = t
        return time_ranges

    for dataset in datasets:
        entity_path =  "../data/" + dataset + "/entity2id.txt"
        entity_list = []
        with open(entity_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                # rel, id = line.strip().split("\t")
                # begin = rel.find('(')
                # w1 = rel[:begin].strip()
                # entity_list.append(w1)
                entity = line.split('\t')[0].strip('\n').strip('<').strip('>').strip('.').replace('\\', '/')
                entity_list.append(entity)
        print(entity_list)
        time_ranges = get_entity_time_ranges(dataset)
        craw = Crawler_google_images()
        for entity in entity_list:
            t_range = time_ranges.get(entity, {})
            min_t = t_range.get('min_t', '')
            max_t = t_range.get('max_t', '')
            if min_t and max_t:
                keyword = entity + f"\t{min_t}-{max_t}"
            else:
                keyword = entity
            url = 'https://www.google.com/search?q=' + entity + '&tbm=isch'
            craw.run(dataset, keyword, url)

