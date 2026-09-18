import json

# 원본 JSONL 파일 경로
input_file = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl"

# 출력 JSONL 파일 경로
output_file = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/raw_links_small_new.jsonl"

# JSONL 파일 분류
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        data = json.loads(line)  # JSONL 한 줄씩 로드
        print(data)
        if "형법 제38조(경합범과 처벌례)" not in data["article2"]:
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")