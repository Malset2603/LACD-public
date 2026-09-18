from src.utils.utils import article_key_function
import pandas as pd

def detect_delete(article_text):
    article_key = article_key_function(article_text)
    # Check if "삭제" is in the article key
    if article_key + " " + "삭제" in article_text:
        return True
    else:
        return False

def detect_civil(article_text):
    if article_text.startswith('민법'):
        return True
    else:
        return False

def detect_combined(article_text):
    # Combine both delete and civil detection
    return detect_delete(article_text) or detect_civil(article_text)

if __name__ == "__main__":
    # Load the JSONL file into a DataFrame
    df = pd.read_json('./data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl', lines=True)

    # Apply the detect_combined function to filter rows
    filtered_df = df[
        ~(df['article1'].apply(detect_combined)) & ~(df['article2'].apply(detect_combined))
    ]

    # Adjust the DataFrame to match the original format
    if 'source_law_id' in filtered_df.columns or 'source_article' in filtered_df.columns or 'target_law_id' in filtered_df.columns or 'target_article' in filtered_df.columns:
        filtered_df = filtered_df[['article1', 'article2', 'answer']]  # Retain only relevant columns

    # Convert answer column to boolean (if needed)
    if filtered_df['answer'].dtype != 'bool':
        filtered_df['answer'] = filtered_df['answer'].astype(bool)

    # Save the refined DataFrame to a new JSONL file
    filtered_df.to_json('./data/datasets/LACD-biclassification/checker-generated/raw_links_small_deleterefine.jsonl', 
                        orient='records', lines=True, force_ascii=False)