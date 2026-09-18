# gold retriever

# csv headers
ARTICLE_TITLE='article_title'
CONTENTS = 'contents'

def article_dictionary_list(file_path = './data/database/laws.csv'):
    import pandas as pd
    global article_df
    
    
    article_df = pd.read_csv(file_path, header=0)
    return 

def gold_retriever(retrieval_article_set):
    global article_df

    return article_df[article_df[ARTICLE_TITLE].isin(retrieval_article_set)].contents.tolist()


if __name__ == "__main__":
    article_dictionary_list()
    retrieved_result = gold_retriever(['민법 제1057조의2', '민법 제1003조', '민법 제1009조', '민법 제1065조', '민법 제1112조', '민법 제1114조', '민법 제1115조', '민법 제1057조'])


    for r in retrieved_result:
        print(r)
    avglen = sum([len(a) for a in retrieved_result])/len(retrieved_result)
    print(avglen)
    
    