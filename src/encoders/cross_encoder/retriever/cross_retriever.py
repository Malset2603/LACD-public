import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm, trange
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
import numpy as np

from src.utils.encoder.utils import MAX_TOKEN_LENGTH
from src.utils.utils import article_key_function

import time

def cross_retriever(query, top_k_articles, cross_encoder_model, tokenizer, article_network:ArticleNetwork, batch_size=64, index_method = "none", query_vector = None):
    """
    Function to use a cross-encoder to distinguish conflicts in top-k retrieved articles using batch processing.

    Major args:
    query (str): The original article to compare against top-k retrieved articles.
    top_k_articles (list): List of top-k retrieved articles.
    model_path (str): Path to the cross-encoder model.
    top_k (int): The number of top articles to return based on contradiction score.
    batch_size (int): The number of articles to process in one batch.

    Returns:
    List of (article, score) where score represents the classification score of contradiction.
    """
    # Load tokenizer and cross-encoder model

    if article_key_function(query) not in article_network.article_key_to_idx.keys():
        article_idx = article_network.add_article_node(query)
    else:
        article_idx = article_network.article_key_to_idx[article_key_function(query)]

    if index_method != "none":
        if isinstance(query_vector, np.ndarray):
            query_vector = torch.tensor(query_vector).to(cross_encoder_model.vector_tensor.device)
        if query_vector.dim() == 1: # type: ignore
            query_vector = query_vector.unsqueeze(0)  # type: ignore # (1, D) 형태로 변환
        cross_encoder_model.vector_tensor = torch.cat([cross_encoder_model.vector_tensor, query_vector], dim=0) # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_encoder_model.to(device)

    retrieval_start_time = time.time()


    # List to store results
    conflicts = []

    # Process top_k_articles in batches
    for i in range(0, len(top_k_articles), batch_size):
        batch_articles = top_k_articles[i:i+batch_size]



        max_length = min([MAX_TOKEN_LENGTH, tokenizer.model_max_length])

        # Tokenize the batch of (article, batch_articles) pairs
        inputs = tokenizer.batch_encode_plus(
            [(query, retrieved_article) for retrieved_article in batch_articles],
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move input tensors to the appropriate device
        input_ids = inputs['input_ids'].to(device) # type: ignore
        attention_mask = inputs['attention_mask'].to(device) # type: ignore



        article2_idx_list = [
            article_network.article_key_to_idx[article_key_function(retrieved_article)]
            for retrieved_article in batch_articles
        ]
        article1_idx_tensor = torch.tensor([article_idx for _ in batch_articles]).to(device)
        article2_idx_tensor = torch.tensor(article2_idx_list).to(device)  # Convert to tensor and move to device


        # Get model prediction for the batch
        with torch.no_grad():
            if index_method != "none":
                outputs = cross_encoder_model(article1_idx=article1_idx_tensor,article2_idx=article2_idx_tensor,
                input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = cross_encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            if logits.shape[1] == 1:
                # 한 개의 확률을 출력하는 경우 (예: [batch_size, 1])
                probs = torch.sigmoid(logits).cpu().numpy()  # Sigmoid로 변환하여 확률 얻기
            else:
                # 두 개의 확률을 출력하는 경우 (예: [batch_size, 2])
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # Softmax로 변환하여 확률 얻기

        # Process the results for each article in the batch
        for idx, retrieved_article in enumerate(batch_articles):
            if logits.shape[1] == 1:
                # Sigmoid로 변환된 단일 값 (True일 확률)
                contradiction_score = probs[idx][0]
            else:
                # Softmax로 변환된 두 번째 값 (True일 확률)
                contradiction_score = probs[idx][1]  # 두 번째 값이 True일 확률
            
            conflicts.append((retrieved_article, contradiction_score))

    # Sort conflicts based on the contradiction score in descending order
    conflicts = sorted(conflicts, key=lambda x: x[1], reverse=True)
    
    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"crossencoder 실행 시간: {elapsed_time:.6f}초")


    return conflicts
