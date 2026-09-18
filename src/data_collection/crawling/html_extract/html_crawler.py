import pandas as pd
from urllib.request import urlopen
from tqdm import trange
from src.preprocessing.utils import articlename_to_article_id
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import quote
import os

# 이 코드는 법률을 HTML 파일 형식으로 크롤링하는 코드입니다.

# law_id 는 법률 고유의 MST 코드
# law_article_num 은 법률의 조문 번호
# law_article_secondnum 은 조의~ 인 경우 붙음
def law_id_to_url(law_id, law_article_num, law_article_secondnum):
    raw_url = "https://www.law.go.kr/LSW//lsSideInfoP.do?&lsiSeq={}&joNo={}&joBrNo={}&docCls=jo&urlMode=lsScJoRltInfoR".format(law_id, str(law_article_num).zfill(4), str(law_article_secondnum).zfill(2))
    return quote(raw_url, safe=":/&?=")

def extract_lawcon_content(html_content):

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, 'html.parser')

    # class가 'lawcon'인 div 태그 추출
    lawcon_div = soup.find('div', class_='lawcon')

    if lawcon_div:
        lawcon_text = lawcon_div.get_text(strip=True)
    else:
        lawcon_text="No <div class='lawcon'> found."

    if lawcon_div:
        # 'onclick' 속성을 가진 모든 태그 찾기
        onclick_tags = lawcon_div.find_all(attrs={'onclick': True})
        
        # 정규 표현식을 사용해 'fncLsLawPop' 호출의 인자만 추출
        fnclsLawPop_args = []
        for tag in onclick_tags:
            onclick_value = tag['onclick']
            # 정규식으로 fncLsLawPop 함수의 첫 번째 인자를 추출
            match = re.search(r"fncLsLawPop\('([^']*)',\s*'JO'", onclick_value)
            if match:
                fnclsLawPop_args.append(match.group(1))  # 첫 번째 인자만 리스트에 추가

        return lawcon_text, fnclsLawPop_args
    else:
        return lawcon_text, []


if __name__ == "__main__":
    
    laws_path = "./data/database/laws_list.csv"
    laws_csv_dir = "./data/database/laws/"
    law_list = pd.read_csv(laws_path, encoding="UTF8")
    
    # 파일이 이미 존재하는 경우 로드해서 중복 체크
    saved_laws = []
    output_file = './data/database/laws_html.jsonl'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                entry = json.loads(line)
                saved_laws.append((entry['law_name'], entry['article_key'], entry['article_num'], entry['article_secondnum']))

    for i in trange(len(law_list)):
        law_contents_list = []

        # row 가 법률
        law_name = law_list.loc[i]["법령명한글"]
        law_id = law_list.loc[i]['법령일련번호']

        article_path = laws_csv_dir+law_name+".csv"
        try:
            law_article_df = pd.read_csv(str(article_path), encoding="UTF8")
        except:
            continue

        for j in range(len(law_article_df)):
            if law_article_df.loc[j]['조문여부'] != '조문':
                continue

            law_article_key = articlename_to_article_id(law_article_df.loc[j]['조문내용'].split("(")[0].split(" ")[0])
            if "-" in law_article_key:
                law_article_num = law_article_key.split("-")[0]
                law_article_secondnum = law_article_key.split("-")[1]
            else:
                law_article_num = law_article_key
                law_article_secondnum = "0"

            # 중복 확인
            if (law_name, law_article_key, law_article_num, law_article_secondnum) in saved_laws:
                continue

            url = law_id_to_url(law_id=law_id, law_article_num=law_article_num, law_article_secondnum=law_article_secondnum)
            try:
                detail = urlopen(url).read()
            except:
                continue

            lawcon_contents, link_ids = extract_lawcon_content(detail)

            instance = {
                "law_name": law_name,
                "article_key": law_article_key,
                "article_num": law_article_num,
                "article_secondnum": law_article_secondnum,
                "article_contents": lawcon_contents,
                "link_ids": link_ids
            }

            law_contents_list.append(instance)
    
        # jsonl 파일에 쓰기
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for entry in law_contents_list:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')
