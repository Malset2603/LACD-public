import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange
import csv
import os

def xml_to_csv_row(contents, csv_writer):
    # XML 데이터를 파싱
    root = ET.fromstring(contents)

    # JoSection 데이터 추출 및 CSV에 쓰기
    for jo in root.findall('JoSection/Jo'):
        jo_no = jo.get('No')
        chap_no = jo.find('chapNo').text # type: ignore
        jo_no_text = jo.find('joNo').text # type: ignore
        jo_br_no = jo.find('joBrNo').text # type: ignore
        jo_yn = jo.find('joYn').text # type: ignore
        jo_ttl = jo.find('joTtl').text if jo.find('joTtl') is not None else '' # type: ignore
        jo_cts = jo.find('joCts').text # type: ignore
        
        csv_writer.writerow([
            'JoSection', jo_no, chap_no, jo_no_text, jo_br_no, jo_yn, jo_ttl, jo_cts, '', '', '', '', '', ''
        ])

    # ArSection 데이터 추출 및 CSV에 쓰기
    for ar in root.findall('ArSection/Ar'):
        ar_no = ar.get('No')
        ar_anc_yd = ar.find('arAncYd').text # type: ignore
        ar_anc_no = ar.find('arAncNo').text # type: ignore
        ar_cts = ar.find('arCts').text # type: ignore
        
        csv_writer.writerow([
            'ArSection', ar_no, '', '', '', '', '', '', ar_anc_yd, ar_anc_no, ar_cts, '', '', ''
        ])

    # BylSection 데이터 추출 및 CSV에 쓰기
    for byl in root.findall('BylSection/Byl'):
        byl_no = byl.get('No')
        byl_anc_yd = byl.find('bylAncYd').text # type: ignore
        byl_anc_no = byl.find('bylAncNo').text # type: ignore
        byl_cts = byl.find('bylCts').text # type: ignore
        
        csv_writer.writerow([
            'BylSection', byl_no, '', '', '', '', '', '', '', '', '', byl_anc_yd, byl_anc_no, byl_cts
        ])

def articles_extract(laws_dir="./data/database/laws_list.csv"):
    # CSV 파일을 불러오는 단계
    law_list = pd.read_csv(laws_dir, encoding="UTF8")
    my_id = None
    base_dir = "./data/database/laws_english/"
    os.makedirs(base_dir, exist_ok=True)

    for i in trange(len(law_list)):
        law_id = law_list.loc[i]['법령ID']
        url = f"http://www.law.go.kr/DRF/lawService.do?OC={my_id}&target=elaw&ID={law_id}&type=XML"
        try:
            contents = urlopen(url).read()

            # 각 법령 ID에 맞는 CSV 파일 생성
            csv_file_path = os.path.join(base_dir, f"{str(law_id).zfill(6)}.csv")
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)

                # CSV 헤더 작성
                csvwriter.writerow([
                    'Section', 'No', 'chapNo', 'joNo', 'joBrNo', 'joYn', 'joTtl', 'joCts', 
                    'arAncYd', 'arAncNo', 'arCts', 
                    'bylAncYd', 'bylAncNo', 'bylCts'
                ])

                # XML 데이터를 CSV에 쓰기
                xml_to_csv_row(contents, csvwriter)
        
        except Exception as e:
            # print(f"Error processing law ID {law_id}: {e}")
            continue

    print("XML 데이터가 개별 CSV 파일로 저장되었습니다.")

# 사용 예시

if __name__ == "__main__":
    articles_extract()
