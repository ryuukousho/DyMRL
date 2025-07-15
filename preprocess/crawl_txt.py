import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
from urllib.parse import quote


class EntityTxtCrawler:
    def __init__(self, dataset, output_dir="../data/"):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.output_dir = output_dir + dataset + "/txt"

        self.date_pattern = re.compile(
            r'(?:\b\d{4}-\d{2}-\d{2}\b)|'
            r'(?:\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b)|'
            r'(?:\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}\b)'
        )

    def get_most_relevant_paragraph(self, entity):
        url = f"https://en.wikipedia.org/wiki/{quote(entity.replace(' ', '_'))}"
        print(f"Processing: {entity}")

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".mw-parser-output p"))
            )

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            for tr in soup.select('.infobox tr'):
                if tr.th and tr.td:
                    text = f"{tr.th.get_text().strip()}: {tr.td.get_text().strip()}"
                    if self.date_pattern.search(text):
                        return text

            for p in soup.select('.mw-parser-output p'):
                text = p.get_text().strip()
                if self.date_pattern.search(text) and len(text) > 20:
                    return text

            for li in soup.select('.timeline li, .timeline-event'):
                text = li.get_text().strip()
                if self.date_pattern.search(text):
                    return text

            return f"No time-sensitive description found for {entity}"

        except Exception as e:
            return f"Error scraping {entity}: {str(e)}"

    def save_description(self, entity, text):
        filepath = os.path.join(self.output_dir, entity)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved: {filepath}")

    def process_entities(self, entities):
        for entity in entities:
            description = self.get_most_relevant_paragraph(entity)
            self.save_description(entity, description)

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    datasets = ["ICE14-IMG-TXT", "ICE0515-IMG-TXT", "ICE18-IMG-TXT", "GDELT-IMG-TXT"]
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

        crawler = EntityTxtCrawler(dataset, output_dir="../data/")
        try:
            crawler.process_entities(entity_list)
        finally:
            crawler.close()