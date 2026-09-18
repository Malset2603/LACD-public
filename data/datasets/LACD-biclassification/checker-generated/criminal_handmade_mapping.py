# csv 파일에 저장된 형법의 법률관계를 적어둔 문서를 보고 pair 를 생성하는 코드.
# gold retriever 가 필요함
# old code 고 이제 사용할 수 없음.

# from telnetlib import NEW_ENVIRON
import pandas as pd
from src.utils.retrieval_methods.gold import *
import json
from src.preprocessing.utils import articlename_to_article_id


if __name__ == "__main__":

    # criminal handmade 를 저장할 장소
    save_path = "./data/datasets/LACD-biclassification/additional_data/criminal_handmade_mapping.jsonl"

    # csv path
    csv_path = "./data/datasets/LACD-biclassification/checker-generated/criminal_handmade_pair.csv"

    law_id_to_name = {"001692": "형법", "001706": "민법"}

    special_law_id_to_name = {
        "255433": "성폭력범죄의 처벌 등에 관한 특례법",
        "199773": "특정경제범죄 가중처벌 등에 관한 법률",
        "252941": "특정범죄 가중처벌 등에 관한 법률",
        "178989": "폭력행위 등 처벌에 관한 법률",
        "249825": "아동ㆍ청소년의 성보호에 관한 법률",
    }

    law_id_to_name = {**law_id_to_name, **special_law_id_to_name}
    law_name_to_id =  {v:k for k,v in law_id_to_name.items()}

    handmade_df = pd.read_csv(csv_path, header = 0)

    article_dictionary_list()
    new_instance_list = []
    for idx, row in handmade_df.iterrows():
        if str(row['article1']) == 'nan':
            continue
        print(row["article1"]+" 제"+row["article1_num"])
        instance = {
            "article1": gold_retriever([row["article1"]+" 제"+row["article1_num"]])[0],
            "article2": gold_retriever([row["article2"]+" 제"+row["article2_num"]])[0],
            "answer": True if (str(row["memo"]) == "nan") or "모순 아님" not in row['memo'] else False,
            "source_law_id": law_name_to_id[row['article1']],
            "source_article": articlename_to_article_id(row["article1_num"]),
            "target_law_id": law_name_to_id[row['article2']],
            "target_article": articlename_to_article_id(row["article2_num"]),
        }
        new_instance_list.append(instance)

    print("total length: ", len(new_instance_list))

    # **여기서 new_instance_list를 article1을 기준으로 정렬합니다.**
    new_instance_list.sort(key=lambda x: x['article1'])

    # 병합된 결과를 새로운 JSONL 파일로 저장
    with open(save_path, 'w', encoding='utf-8') as outfile:
        for item in new_instance_list:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')