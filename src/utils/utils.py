SEED = 42

def article_key_function(text)->str:
    import re
    
    # Regular expression to match "제O조(의O)"
    text = text.replace("·", "ㆍ")
    match = re.search(r'제[\d]+조(?:의[\d]+)?', text)
    if match:
        return text[:match.end()]
    else:
        print("text is:", text)
        assert(0)
        return "None"
    
def seed_everything(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stratified_mini_sample(df, ratio, seed=42, label_col="answer"):
    """Stratified mini sampling: preserves P(y=1) ratio. ratio in (0,1]."""
    import pandas as pd
    if ratio is None or ratio >= 1.0:
        return df
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"mini_ratio must be in (0,1], got {ratio}")
    # try stratified per label; fallback to random if single class
    try:
        from sklearn.model_selection import train_test_split
        n_total = len(df)
        # use train_test_split with stratify to get exactly ratio
        if df[label_col].nunique() < 2:
            return df.sample(frac=ratio, random_state=seed).reset_index(drop=True)
        # stratify requires at least 2 samples per class after split; guard small df
        min_class = df[label_col].value_counts().min()
        if min_class * ratio < 1:
            # fallback to per-class frac sampling
            sampled = []
            for _, group in df.groupby(label_col):
                k = max(1, int(len(group) * ratio))
                sampled.append(group.sample(n=k, random_state=seed))
            import pandas as _pd2
            return _pd2.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)
        sampled_df, _ = train_test_split(df, train_size=ratio, stratify=df[label_col], random_state=seed)
        return sampled_df.reset_index(drop=True)
    except ImportError:
        # sklearn not available: manual per-class sampling
        import pandas as _pd3
        sampled = []
        for _, group in df.groupby(label_col):
            k = max(1, int(len(group) * ratio))
            sampled.append(group.sample(n=k, random_state=seed))
        return _pd3.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

def mini_rows_sample(rows, ratio, seed=42, label_key="answer"):
    """Stratified sample for list[dict] with label_key. Preserves label distribution."""
    if ratio is None or ratio >= 1.0:
        return rows
    import random
    import collections
    random.seed(seed)
    # group by label
    groups = collections.defaultdict(list)
    for r in rows:
        groups[bool(r.get(label_key, False))].append(r)
    sampled = []
    for label, grp in groups.items():
        k = max(1, int(len(grp) * ratio))
        # deterministic shuffle then take k
        rnd = random.Random(seed + int(label))
        grp_shuffled = grp.copy()
        rnd.shuffle(grp_shuffled)
        sampled.extend(grp_shuffled[:k])
    rnd_all = random.Random(seed)
    rnd_all.shuffle(sampled)
    return sampled

import os
import json

def process_jsonl_file(file_path):
    """JSONL 파일을 읽고, article_key_function을 적용한 후 다시 저장"""
    new_jsonl_list = []
    
    # JSONL 파일 읽기
    with open(file_path, mode='r', encoding='utf-8') as jsonlfile:
        for line in jsonlfile:
            row = json.loads(line)
            if 'article_key' in row and row['article_key'] != "":
                try:
                    row['article_key'] = article_key_function(row['article'])
                except:
                    print(row)
                    exit(0)
            new_jsonl_list.append(row)
    
    # JSONL 파일에 다시 저장
    with open(file_path, mode='w', encoding='utf-8') as jsonlfile:
        for row in new_jsonl_list:
            jsonlfile.write(json.dumps(row) + '\n')
import json

def unescape_unicode_in_jsonl(file_path):
    # 파일의 모든 내용을 메모리에 저장한 후 다시 파일에 덮어쓰기
    unescaped_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # JSONL 파일에서 한 줄씩 읽고 JSON 파싱
            row = json.loads(line.strip())
            
            # 모든 컬럼의 값들을 유니코드 escape에서 UTF-8로 변환
            unescaped_row = {key: value.encode('utf-8').decode('unicode_escape') for key, value in row.items()}
            
            # 변환된 row를 다시 JSON 문자열로 변환하여 저장
            unescaped_lines.append(json.dumps(unescaped_row, ensure_ascii=False))
    
    # 변환된 내용을 원본 파일에 덮어쓰기
    with open(file_path, 'w', encoding='utf-8') as f:
        for unescaped_line in unescaped_lines:
            f.write(unescaped_line + '\n')

def process_directory(directory_path):
    """지정된 디렉토리와 하위 디렉토리에서 모든 JSONL 파일을 찾아 처리"""
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                unescape_unicode_in_jsonl(file_path)

if __name__ == "__main__":
    directory_path = "./data/database/generated_case_cache"
    process_directory(directory_path)