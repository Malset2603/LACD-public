# laws 아래 모든 csv 파일에 대해서 하나의 laws.csv 파일로 옮기는 코드
import os
import glob
import pandas as pd
import ast

# from sympy import content
from src.data_collection.preprocessing.utils import recursive_data_extract
from tqdm import tqdm
from src.utils.utils import article_key_function


if __name__ == "__main__":
    laws_dir = "./data/database/laws/"
    
    # csv 파일의 경로를 모두 가져옴
    csv_files = glob.glob(os.path.join(laws_dir, "*.csv"))
    
    instances = []

    # 각 csv 파일을 순회
    for csv_file in tqdm(csv_files):
        # pandas로 csv 파일을 읽어들임
        df = pd.read_csv(csv_file)
        # 각 행을 순회
        for index, row in df.iterrows():
            row_dict = row.to_dict()

            # 조문번호가 존재하는지 확인
            if row_dict['조문여부'] != "조문":
                continue
            
            # 법률 이름과 조문 정보 구성
            law_name = os.path.basename(csv_file).split(".csv")[0]  # 파일명에서 법률 이름 추출
            article_id = row_dict['조문번호']  # 따옴표 없이 정수형으로 처리
            article_title = article_key_function(law_name + " " + row_dict['조문내용'])
            
            if '항' in row_dict.keys() and str(row_dict['항']) != "nan":
                contents = law_name + row_dict['조문내용'] + recursive_data_extract(ast.literal_eval(row_dict['항']))
            else:
                contents = law_name + row_dict['조문내용']
            
            contents.replace('\n', '')


            # 인스턴스 추가
            instance = {
                "law_name": law_name,
                "article_id": article_id,  # article_id는 문자열이 아닌 그대로 둠
                "article_title": article_title,
                "contents": contents,
            }

            instances.append(instance)  # instances 리스트에 추가



    
    # 모든 인스턴스를 DataFrame으로 변환
    instances_df = pd.DataFrame(instances)

    # DataFrame을 하나의 CSV 파일로 저장, article_id는 따옴표 없이 저장되도록 QUOTE_MINIMAL 사용
    output_file = "./data/database/laws_20240930.csv"
    instances_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"All data saved to {output_file}")
