import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.utils import seed_everything

def balance_dataset(df, label_col='answer', ratio=10):
    true_df = df[df[label_col] == True]
    false_df = df[df[label_col] == False]
    false_df = false_df.sample(frac=1).reset_index(drop=True)
    
    max_false_size = min(len(false_df), len(true_df) * ratio)
    false_df = false_df.iloc[:max_false_size]
    
    balanced_df = pd.concat([true_df, false_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

if __name__ == "__main__":
    seed_everything(42)
    df = pd.read_json('./data/datasets/LACD-biclassification/checker-generated/raw_links_punishmentrefine.jsonl', lines=True)
    
    # 데이터 균형 맞추기
    df = balance_dataset(df, label_col='answer', ratio=20)
    
    # 먼저 train과 (test + val)을 6:4로 나눔
    train_df, test_val_df = train_test_split(df, test_size=0.4, random_state=42)

    # 나머지 test_val_df를 다시 5:5 비율로 나눠 test와 val을 만듦
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    # 데이터셋을 파일로 저장
    train_df.to_json('./data/datasets/LACD-biclassification/train-test-divide-refine/train.jsonl', orient='records', lines=True, force_ascii=False)
    test_df.to_json('./data/datasets/LACD-biclassification/train-test-divide-refine/test.jsonl', orient='records', lines=True, force_ascii=False)
    val_df.to_json('./data/datasets/LACD-biclassification/train-test-divide-refine/val.jsonl', orient='records', lines=True, force_ascii=False)

    # 데이터셋의 크기를 출력하여 비율이 잘 나누어졌는지 확인
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Validation set size: {len(val_df)}")
