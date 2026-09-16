import json
from src.utils.utils import article_key_function

# 파일 읽기
with open('missed_pairs.jsonl', 'r', encoding='utf-8') as file:
    lines = [json.loads(line) for line in file]

# 중복되는 article1과 article2를 찾고, answer가 None인 instance 제거
filtered_lines = []
seen_pairs = set()

for line in lines:
    article1 = line['article1']
    article2 = line['article2']
    answer = line.get('answer')

    pair = (article_key_function(article1), article_key_function(article2))
    reversed_pair = (article_key_function(article2), article_key_function(article1))

    # 중복되는 pair 확인
    if pair in seen_pairs or reversed_pair in seen_pairs:
        # answer가 None이 아닌 경우에만 유지
        if answer is not None:
            filtered_lines.append(line)
    else:
        filtered_lines.append(line)
        seen_pairs.add(pair)

# 파일 저장
with open('missed_pairs_refine.jsonl', 'w', encoding='utf-8') as file:
    for line in filtered_lines:
        file.write(json.dumps(line, ensure_ascii=False) + '\n')
print(len(filtered_lines))