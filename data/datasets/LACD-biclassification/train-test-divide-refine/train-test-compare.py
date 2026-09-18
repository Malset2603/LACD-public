import json
from collections import Counter
from tqdm import tqdm
import sys

from json.decoder import JSONDecodeError, JSONDecoder


from src.utils.utils import article_key_function

decoder = JSONDecoder()


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                obj, _ = decoder.raw_decode(line)
            except JSONDecodeError as e:
                print(f"[{split}] Skipping invalid JSON on line {lineno}: {e}")
                continue
            data.append(obj)
    return data

if __name__ == "__main__":

    query_path = "./data/datasets/LACD-retrieval/queries.jsonl"
    query_data = read_jsonl(query_path)
    
    query_data = set([article_key_function(q["query"]) for q in query_data[:]])
    
    # File paths
    file_paths = ["./data/datasets/LACD-biclassification/train-test-divide-refine/"+p for p in [ "train.jsonl", "val.jsonl"]]
    # file_paths.append("/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-retrieval/missed_pairs.jsonl")

    # test 에 있는 것들이 train 이나 val 에 있는지 확인
    test_data = read_jsonl("./data/datasets/LACD-biclassification/train-test-divide-refine/test.jsonl")
    test_data = [(article_key_function(row["article1"]), article_key_function(row["article2"])) for row in test_data if row["answer"]]

    # Process each file to extract the specified elements
    for file_path in file_paths:
        count = 0
        data = read_jsonl(file_path)
        for row in tqdm(data):
            if (
                (article_key_function(row["article1"]), article_key_function(row["article2"])) in test_data
                or
                (article_key_function(row["article2"]), article_key_function(row["article1"])) in test_data
            ):
                count += 1
                # print(row)

        print(file_path)
        print(count)
    # print(not_in_query_data)