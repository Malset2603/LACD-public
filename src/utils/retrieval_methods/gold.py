# gold retriever

# csv headers
ARTICLE_TITLE='article_title'
CONTENTS = 'contents'

def article_dictionary_list(file_path = './data/database/laws.csv'):
    import pandas as pd
    global article_df, title_to_content



    article_df = pd.read_csv(file_path, header=0)
    # O(1) title -> contents index built once (raw titles as keys to preserve
    # exact match semantics; first-wins mirrors dataframe order on duplicates).
    title_to_content = {}
    for t, c in zip(article_df[ARTICLE_TITLE].tolist(), article_df[CONTENTS].tolist()):
        title_to_content.setdefault(t, c)
    return

def gold_retriever(retrieval_article_set):
    global title_to_content

    return [title_to_content[t] for t in retrieval_article_set if t in title_to_content]


if __name__ == "__main__":
    article_dictionary_list()
    retrieved_result = gold_retriever(['민법 제1057조의2', '민법 제1003조', '민법 제1009조', '민법 제1065조', '민법 제1112조', '민법 제1114조', '민법 제1115조', '민법 제1057조'])


    for r in retrieved_result:
        print(r)
    avglen = sum([len(a) for a in retrieved_result])/len(retrieved_result)
    print(avglen)
    
    