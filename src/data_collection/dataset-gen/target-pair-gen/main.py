import pandas as pd
import json
from tqdm import tqdm
from src.preprocessing.utils import lawname_to_lawid, articlename_to_article_id, generate_law_id_pair_list

if __name__ == "__main__":
    from src.utils.retrieval_methods.gold import article_dictionary_list, gold_retriever

    # Configure retriever
    article_df = article_dictionary_list()
    retriever = gold_retriever

    law_list = generate_law_id_pair_list()


    # Read the CSV file
    law_link_df = pd.read_csv('./data/database/law_link.csv')

    # Filter rows where category is "other law"
    filtered_rows = law_link_df[law_link_df['category'] == 'other law']

    # Prepare data for JSONL file
    jsonl_data = []
    for _, row in tqdm(filtered_rows.iterrows(), total=filtered_rows.shape[0], desc="Processing rows", unit="row"):
        if row["source_law"] in {"형법", "민법"} or row["target_law"] in {"형법", "민법"}: 
           pass
        else:
            continue 
        
        source_retrieval = ["{} {}".format(row["source_law"],row["source_article"])]
        target_retrieval = ["{} {}".format(row["target_law"],row["target_article"])]
        


        try:
            src_article= retriever(source_retrieval)[0]
            target_article = retriever(target_retrieval)[0]
        except:
            continue
        # exit()

        data_entry = {
            "article1": src_article,
            "article2": target_article,
            "answer": None,
            "source_law_id": lawname_to_lawid(law_list, row["source_law"]),
            "source_article": articlename_to_article_id(row["source_article"]),
            "target_law_id": lawname_to_lawid(law_list, row["target_law"]),
            "target_article": articlename_to_article_id(row["target_article"])
        }
        jsonl_data.append(data_entry)

    # Write to JSONL file
    with open('./data/datasets/LACD-biclassification/checker-generated/raw_links_small_renew.jsonl', 'w', encoding='utf-8') as outfile:
        for entry in jsonl_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

