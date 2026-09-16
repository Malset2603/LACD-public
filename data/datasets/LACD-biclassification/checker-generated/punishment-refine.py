import json

# 원본 JSONL 파일 경로
input_file = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/true_answers_original.jsonl"

# 출력 JSONL 파일 경로
true_output_file = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/true_answers.jsonl"
false_output_file = "/mnt/disk1/anseon2001/LACD/data/datasets/LACD-biclassification/checker-generated/false_answers.jsonl"

# JSONL 파일 분류
with open(input_file, "r", encoding="utf-8") as infile, \
     open(true_output_file, "w", encoding="utf-8") as true_out, \
     open(false_output_file, "w", encoding="utf-8") as false_out:

    for line in infile:
        data = json.loads(line)  # JSONL 한 줄씩 로드
        if data.get("answer") is True:
            true_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        elif data.get("answer") is False:
            false_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ JSONL 파일이 성공적으로 분리되었습니다!\n- True 데이터: {true_output_file}\n- False 데이터: {false_output_file}")