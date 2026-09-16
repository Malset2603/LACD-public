import pandas as pd
from urllib.request import urlopen, Request
from tqdm import tqdm
# from src.preprocessing.utils import articlename_to_article_id
from src.utils.utils import article_key_function
from bs4 import BeautifulSoup
import json
from urllib.parse import quote
import time
import random

# Random 딜레이 범위 설정 (예: 1초에서 3초 사이)
MIN_DELAY = 0.01
MAX_DELAY = 0.05

# User-Agent 설정 (브라우저처럼 위장)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 법령 ID를 기반으로 URL 생성
def law_id_to_url(link_id):
    raw_url = "https://www.law.go.kr/LSW//lsLinkCommonInfo.do?lsJoLnkSeq={}&chrClsCd=010202&ancYnChk=".format(link_id)
    return quote(raw_url, safe=":/&?=")

# HTML에서 법령명과 lawcon 내용을 추출하는 함수
def extract_lawcon_content(html_content):

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, 'html.parser')

    # 법령명 추출 (class가 'cont_top'인 div 내의 h2 태그)
    cont_top = soup.find('div', class_='cont_top')
    law_name = cont_top.find('h2') if cont_top else None
    
    if law_name:
        # span 태그와 그 내용을 제거
        for span in law_name.find_all('span'):
            span.extract()
        law_name_text = law_name.get_text(strip=True) 
    else:
        law_name_text = "No law name found."

    # class가 'lawcon'인 모든 div 태그 추출
    lawcon_divs = soup.find_all('div', class_='lawcon')

    # 모든 lawcon div 태그의 텍스트를 리스트로 저장
    lawcon_texts = [div.get_text(strip=True) for div in lawcon_divs]

    if not lawcon_texts:
        lawcon_texts = ["No <div class='lawcon'> found."]

    return law_name_text, lawcon_texts

# row를 받아서 link instance를 retrieve 하는 함수
def link_retriever(row):
    
    if len(row["link_ids"]) == 0:
        return []
    
    source_article_key = row["law_name"] + " " + str("제{}조".format(row['article_key']) if "-" not in row['article_key'] else "제{}조의{}".format(row['article_num'], row['article_secondnum']))
    instances = []

    for id in row["link_ids"]:
        url = law_id_to_url(id)

        # User-Agent를 포함한 요청 만들기
        req = Request(url, headers=headers)

        # 요청 전 랜덤한 딜레이 추가
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        # URL 열기 및 내용 읽기
        detail = urlopen(req).read()

        law_name, lawcon_contents = extract_lawcon_content(detail)
        
        for c in lawcon_contents:
            if c == "No <div class='lawcon'> found.":
                continue
            target_article_key = article_key_function(law_name + " " + c)
            instances.append({'source_key': source_article_key, "target_key": target_article_key})

    return instances

if __name__ == "__main__":
    
    laws_path = "./data/database/laws_html.jsonl"
    save_path = "./data/database/law_link_20240930_supplymentary_110000.jsonl"

    law_data = []
    with open(laws_path, 'r', encoding='utf-8') as file:
        for line in file:
            law_data.append(json.loads(line.strip()))

    # 총 몇 개가 이미 저장되었는가?
    # saved_source_key = set()
    # with open(save_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         saved_source_key = saved_source_key.add()

    law_data = [row for row in law_data if len(row['link_ids'])!= 0]

    link_instances = []
    for idx, row in enumerate(tqdm(law_data)):
        # 이미 저장된 데이터는 건너뛰기
        if idx < 110000:
            continue
        link_instances = link_instances + link_retriever(row)

        # jsonl 파일에 쓰기
        if idx % 10000 == 0:
            with open(save_path, 'a', encoding='utf-8') as outfile:
                for entry in link_instances:
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')

            print(idx)
            

            link_instances = []
    
    with open(save_path, 'a', encoding='utf-8') as outfile:
        for entry in link_instances:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
