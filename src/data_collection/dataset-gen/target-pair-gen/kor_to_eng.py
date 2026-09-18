
import pandas as pd

def kor_to_eng_law(law_id, article_id, eng_law_path="./data/database/laws_english"):
    law_path = eng_law_path + f"/{law_id}.csv"
    
    df = pd.read_csv(law_path)
    
    # Check if "joCts" column exists
    if "joCts" not in df.columns:
        raise ValueError("The CSV file does not contain a 'joCts' column")
    
    # Drop or fill NA/NaN values in "joCts" column
    df = df.dropna(subset=["joCts"])
    
    # Filter rows where "joCts" starts with the specified article
    result = df[df["joCts"].str.startswith(f"Article {article_id}")]
    
    # Ensure that the result is unique and return it
    if len(result) != 1:
        raise ValueError(f"Expected exactly one result, but found {len(result)}")
    
    return str(result["joCts"].tolist()[0])

if __name__ == "__main__":
    import json
    import tqdm
    jsonl_file_path = "./data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl"
    save_file_path = "./data/datasets/LACD-biclassification/checker-generated_en/raw_links_small.jsonl"
    english_row_list = []
    with open(jsonl_file_path, 'r', encoding= 'utf-8') as file:
        for line in tqdm.tqdm(file):
            record = json.loads(line)
            try:
                tar_article = kor_to_eng_law(record['target_law_id'], record['target_article'])
                record['b'] = tar_article
                src_article = kor_to_eng_law(record['source_law_id'], record['source_article'])
                record['a'] = src_article
            except:
                continue
            # print(record)
            english_row_list.append(record)
    
    print(f'english row length: {len(english_row_list)}')
    with open(save_file_path, 'w', encoding='utf-8') as file:
        for row in english_row_list:
            json_line = json.dumps(row)
            file.write(json_line + '\n')
    
