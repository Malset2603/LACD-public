import random
import pandas as pd
import ast
import json
import tqdm
from src.data_collection.preprocessing.utils import articlename_to_article_id, recursive_data_extract

if __name__ == "__main__":

    random.seed(42)
    # 법률 파일 오픈
    df_criminal = pd.read_csv("./data/database/laws/형법.csv", encoding='utf-8')
    df_civil = pd.read_csv("./data/database/laws/민법.csv", encoding='utf-8')

    law_id_to_df = {"001692": df_criminal, "001706": df_civil}
    law_id_to_name = {"001692": "형법", "001706": "민법"}

    special_law_id_to_name = {
        "255433": "성폭력범죄의 처벌 등에 관한 특례법",
        "199773": "특정경제범죄 가중처벌 등에 관한 법률",
        "252941": "특정범죄 가중처벌 등에 관한 법률",
        "178989": "폭력행위 등 처벌에 관한 법률",
        "261453": "아동ㆍ청소년의 성보호에 관한 법률",
    }

    law_id_to_name = {**law_id_to_name, **special_law_id_to_name}

    true_article_count = dict()
    code_contents_dict = dict()

    # 각칙이 여기서 시작함
    df_criminal = df_criminal[df_criminal['조문번호'] >= 87]
    df_criminal['law_id'] = "001692"

    # 특별법도 더해야 함.
    for special_law_id in special_law_id_to_name.keys():
        temp_df = pd.read_csv(f"./data/database/laws/{special_law_id_to_name[special_law_id]}.csv", encoding='utf-8')
        temp_df['law_id'] = special_law_id
        df_criminal = pd.concat([df_criminal, temp_df], axis=0)
    df_criminal = df_criminal[df_criminal['조문여부'] == "조문"]

    pairs_of_criminal_act = set()

    # 뽑기
    k = 1000
    while len(pairs_of_criminal_act) < k:
        pair = tuple(sorted(random.sample(range(len(df_criminal)), 2)))
        pairs_of_criminal_act.add(pair)

    row_pairs = [(df_criminal.iloc[pair[0]], df_criminal.iloc[pair[1]]) for pair in pairs_of_criminal_act]

    new_instance_list = []
    for p in tqdm.tqdm(row_pairs):

        source_article = p[0]

        if str(source_article['항']) != "nan":
            source_article_contents = law_id_to_name[source_article["law_id"]] + " " + source_article['조문내용'] + recursive_data_extract(ast.literal_eval(source_article['항']))
        else:
            source_article_contents = law_id_to_name[source_article["law_id"]] + " " + source_article['조문내용']

        target_article = p[1]

        if str(target_article['항']) != "nan":
            target_article_contents = law_id_to_name[target_article["law_id"]] + " " + target_article['조문내용'] + recursive_data_extract(ast.literal_eval(target_article['항']))
        else:
            target_article_contents = law_id_to_name[target_article["law_id"]] + " " + target_article['조문내용']

        instance = {
            "article1": source_article_contents,
            "article2": target_article_contents,
            "answer": False,
            "source_law_id": source_article["law_id"],
            "source_article": articlename_to_article_id(source_article['조문내용'].split("(")[0]),
            "target_law_id": target_article["law_id"],
            "target_article": articlename_to_article_id(target_article['조문내용'].split("(")[0])
        }

        new_instance_list.append(instance)

    print("total length: ", len(new_instance_list))

    # **여기서 new_instance_list를 article1을 기준으로 정렬합니다.**
    new_instance_list.sort(key=lambda x: x['article1'])

    # 병합된 결과를 새로운 JSONL 파일로 저장
    with open('./data/datasets/LACD-biclassification/additional_data/raw_links_small_hardpair.jsonl', 'w', encoding='utf-8') as outfile:
        for item in new_instance_list:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')