
import json
from src.utils.utils import article_key_function

# 기존 queries.jsonl 파일에서 쿼리 읽기
existing_queries = set()
with open('/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-retrieval/queries.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        existing_queries.add(article_key_function(data['query']))

# train.jsonl 파일에서 answer가 True인 article pair 수집
article_pairs = set()
with open('/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-biclassification/train-test-divide-refine/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        pair = (data['article1'],data['article2'])
        article_pairs.add(pair)

article_cnt = 0
# 새로운 쿼리 생성 및 저장
with open('/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-biclassification/train-test-divide-refine/train-queries.jsonl', 'w') as f:
    for article1, article2 in article_pairs:
        if article_key_function(article1) not in existing_queries:
            query = {
                'idx': article_cnt,
                'query': article1
            }
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
            article_cnt += 1
            existing_queries.add(article_key_function(article1))


        if article_key_function(article2) not in existing_queries:
            query = {
                'idx': article_cnt,
                'query': article2
            }
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
            article_cnt += 1
            existing_queries.add(article_key_function(article2))