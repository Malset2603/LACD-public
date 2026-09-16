import torch
from transformers import AutoTokenizer
import pandas as pd
import chromadb

from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

# from src.utils.encoder.utils import MAX_TOKEN_LENGTH
import time

MAX_TOKEN_LENGTH = 4096
# MAX_TOKEN_LENGTH = 512

# our original code, bi-encoder.
def find_top_conflicts(article, model, tokenizer, chroma_collection, top_k=10, index_method = "none"):
    """
    Function to find top conflicts using cosine similarity method with optional case augmentation.
    
    Args:
    article (str): The input article to check.
    model (torch.nn.Module): Pre-trained model.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_collection: Chroma DB collection containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """
    if top_k > 50000:
        top_k = 500

    # Encode the input article
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    with torch.no_grad():
        encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoded_article.last_hidden_state[0, 0, :].cpu().numpy()

    retrieval_start_time = time.time()

    results = chroma_collection.query(
        query_embeddings=[pooled_output.tolist()],
        n_results=top_k,
        include=["embeddings", "documents"]
    )
    top_k_articles = results["documents"][0]


    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time

    return pooled_output, top_k_articles


def binary_retriever(model, laws_df, chroma_collection, article_to_check, top_k=500, batch_size = 8, tokenizer = None):
    """
    Bi-encoder retriever function to find conflicts in legal texts.

    Args:
    model_path (str): Path to the trained model directory.
    laws_csv_path (str): Path to the laws.csv file.
    chroma_db_name (str): Name of the Chroma DB where encodings will be stored.
    article_to_check (str): The article to check for conflicts.
    classification_method (str): Method to use for classification ('cosine' or 'classification_model').
    top_k_val (int): Number of top-k articles to retrieve.

    Returns:
    List of top-k articles that contradict the input article.
    """


    # 인코딩이 없으면 새로 인코딩하여 저장
    if chroma_collection.count() == 0:
        
        ids = []
        documents = []
        embeddings = []

        idx = 0
        all_articles = []

        for _, row in tqdm(laws_df.iterrows()):
            article = row["contents"]

    
            all_articles.append(article)
            ids.append(str(idx))
            idx += 1

        max_length = min(MAX_TOKEN_LENGTH, tokenizer.model_max_length)

        # Process in batches
        for batch_start in tqdm(range(0, len(all_articles), batch_size)):
            batch_articles = all_articles[batch_start:batch_start + batch_size]
            batch_inputs = tokenizer(batch_articles, add_special_tokens=True, max_length=max_length, 
                                     padding="max_length", truncation=True, return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(model.encoder.device)
            attention_mask = batch_inputs["attention_mask"].to(model.encoder.device)

            with torch.no_grad():
                encoder_to_use = model.passage_encoder if hasattr(model, "passage_encoder") else model.encoder
                encoded_articles = encoder_to_use(input_ids=input_ids, attention_mask=attention_mask)
                pooled_outputs = encoded_articles.last_hidden_state[:, 0, :].cpu().numpy()

            # Collect embeddings and documents
            embeddings.extend(pooled_outputs)
            documents.extend(batch_articles)

        chunk_size = 1024
        total_chunks = len(embeddings) // chunk_size + 1
        embeddings = [e.flatten().tolist() for e in embeddings]

        for chunk_idx in tqdm(range(total_chunks)):
            start_idx = chunk_idx * chunk_size
            end_idx = (chunk_idx + 1) * chunk_size

            chunk_embeddings = embeddings[start_idx:end_idx]
            chunk_ids = ids[start_idx:end_idx]
            chunk_docs = documents[start_idx:end_idx]

            chroma_collection.add(
                documents=chunk_docs,
                embeddings=chunk_embeddings,
                ids=chunk_ids,
            )



    # 특정 article을 입력하여 모순된 법률 찾기

    article_vector, top_conflicts = find_top_conflicts(article_to_check, model, tokenizer, chroma_collection, top_k=top_k)


    return article_vector, top_conflicts


