# Here are how /mnt/disk1/anseon2001/LACD/data/database/laws.csv looks like

# law_name,article_id,article_title,contents
# 국가교육위원회 설치 및 운영에 관한 법률,1,국가교육위원회 설치 및 운영에 관한 법률 제1조,국가교육위원회 설치 및 운영에 관한 법률 제1조(목적) 이 법은 국가교육위원회를 설치하여 교육정책이 사회적 합의에 기반하여 안정적이고 일관되게 추진되도록 함으로써 교육의 자주성ㆍ전문성 및 정치적 중립성을 확보하고 교육발전에 이바지함을 목적으로 한다.
# 국가교육위원회 설치 및 운영에 관한 법률,2,국가교육위원회 설치 및 운영에 관한 법률 제2조,"국가교육위원회 설치 및 운영에 관한 법률 제2조(국가교육위원회의 설치)① 사회적 합의에 기반한 교육비전, 중장기 정책 방향 및 교육제도 개선 등에 관한 국가교육발전계획 수립, 교육정책에 대한 국민의견 수렴ㆍ조정 등에 관한 업무를 수행하기 위하여 대통령 소속으로 국가교육위원회(이하 \""위원회\""라 한다)를 둔다.② 위원회는 그 소관에 속하는 업무를 독립하여 수행한다
# ...

# save it as corpus.jsonl by following format
# {"_id": artile_title, "text": contents}

import csv
import json
from src.utils.utils import article_key_function
from src.utils.retrieval_methods.gold import gold_retriever, article_dictionary_list

def load_jsonl(file_path):
    """
    jsonl 파일을 읽어 각 줄을 JSON 객체로 반환합니다.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def true_dict_gen():
    raw_links_pairs = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl"
    missed_data_path = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-retrieval/missed_pairs.jsonl"
    rows = load_jsonl(raw_links_pairs) + load_jsonl(missed_data_path)

    # article key를 생성하여 일관된 키 값 사용
    rows = [
        {
            "article1": article_key_function(r["article1"]),
            "article2": article_key_function(r["article2"]),
            "answer": r["answer"]
        }
        for r in rows
    ]
    
    # 정답(모순 관계)인 row만 선택
    true_rows = [r for r in rows if r["answer"] is True]

    # 각 기사에 대해 모순 관계에 있는 다른 기사들의 리스트 구성
    true_dicts = {}
    for r in true_rows:
        a1 = r["article1"]
        a2 = r["article2"]
        true_dicts.setdefault(a1, []).append(a2)
        true_dicts.setdefault(a2, []).append(a1)

    return true_dicts



if __name__ == "__main__":
    # Path to the input CSV file
    csv_file_path = "/mnt/disk1/anseon2001/LACD/data/database/laws.csv"

    # Path to the output JSONL file
    jsonl_file_path = "/mnt/disk1/anseon2001/LACD/data/datasets/beir-like-format/corpus.jsonl"

    with open(csv_file_path, "r", encoding="utf-8") as csvfile, open(jsonl_file_path, "w", encoding="utf-8") as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Create the output object with _id as article_title and text as contents
            output = {
                "_id": row["article_title"],
                "text": row["contents"]
            }
            jsonlfile.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Saved corpus to {jsonl_file_path}")

    true_dicts = true_dict_gen()

    # here are how dataset looks like
    # {"article1":"형법 제195조(수도불통) 공중이 먹는 물을 공급하는 수도 그 밖의 시설을 손괴하거나 그 밖의 방법으로 불통(不通)하게 한 자는 1년 이상 10년 이하의 징역에 처한다.","article2":"형법 제193조(수돗물의 사용방해)① 수도(水道)를 통해 공중이 먹는 물로 사용하는 물 또는 그 수원(水原)에 오물을 넣어 먹는 물로 쓰지 못하게 한 자는 1년 이상 10년 이하의 징역에 처한다.② 제1항의 먹는 물 또는 수원에 독물 그 밖에 건강을 해하는 물질을 넣은 자는 2년 이상의 유기징역에 처한다.\n","answer":false}

    # there are one pre-defined function named article_key_function

    # if article1 or article2 in test rows are the key for true_dicts,
    # add article to queries list. (e.g, add "형법 제195조(수도불통) 공중이 먹는 물을 공급하는 수도 그 밖의 시설을 손괴하거나 그 밖의 방법으로 불통(不通)하게 한 자는 1년 이상 10년 이하의 징역에 처한다.")

    # then, construct queries.jsonl file as follows:
    # for q in queries:
    #   {"_id": article_key_function(q), "text":q}

    # also, construct test.tsv as follows:

    # for all items in true_dicts
    # query-id  corpus-id   score
    # key-id    value-id(one-by-one)    1
    # ...

    import json
    article_dictionary_list()

    # --- Load true_dicts ---
    # Here we assume true_dicts is stored in a JSON file. Adjust the path as needed.
    # The structure of true_dicts is assumed to be:
    # { key_article_text: [corpus_id1, corpus_id2, ...], ... }


    # --- Read test rows from test.jsonl ---
    test_rows = []
    with open("/mnt/disk1/anseon2001/LACD/outputs/retrieval_results/article_key/bm25.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    # --- Build queries list ---
    queries = []
    for row in test_rows:
        queries.append(article_key_function(row.get("article_to_check")))

    # Remove duplicates (if needed)
    print(len(queries))
    queries = list(set(queries))
    raw_queries = gold_retriever(queries)

    # --- Save queries.jsonl ---
    with open("/mnt/disk1/anseon2001/LACD/data/datasets/beir-like-format/queries.jsonl", "w", encoding="utf-8") as f:
        for q in raw_queries:
            record = {"_id": article_key_function(q), "text": q}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(raw_queries)} queries to queries.jsonl")

    # --- Construct test.tsv ---
    # For every item in true_dicts, write one line per corpus id (value) with a score of 1.
    with open("/mnt/disk1/anseon2001/LACD/data/datasets/beir-like-format/qrels/test.tsv", "w", encoding="utf-8") as f:
        # If a header is desired, uncomment the following line:
        # f.write("query-id\tcorpus-id\tscore\n")
        for key, corpus_ids in true_dicts.items():
            for corpus_id in corpus_ids:
                # assert(key in queries)

                if key not in queries:
                    continue
                f.write(f"{key}\t{corpus_id}\t1\n")
    print("Saved test.tsv")