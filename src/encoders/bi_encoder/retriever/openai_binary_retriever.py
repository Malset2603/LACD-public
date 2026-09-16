import torch
from transformers import AutoTokenizer
import pandas as pd
import chromadb

from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm


# Helper function for finding conflicts using cosine similarity
def find_top_conflicts(article, openai_client, chroma_collection, top_k=10):
    """
    Function to find top conflicts using cosine similarity with optional case augmentation.
    
    Args:
    article (str): The input article to check.
    openai_client: Initialized OpenAI client for embedding generation.
    chroma_collection: Chroma DB collection containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """


    # Get the embedding for the input article using OpenAI API
    response = openai_client.embeddings.create(input=article, model="text-embedding-3-small")
    pooled_output = response.data[0].embedding

    # Retrieve encoded laws from Chroma DB
    encoded_laws = chroma_collection.get(include=["embeddings", "documents"])
    law_vectors = np.array(encoded_laws["embeddings"])  # Convert stored embeddings to numpy array
    law_documents = encoded_laws["documents"]

    similarities = cosine_similarity(
        torch.tensor(pooled_output).unsqueeze(0),
        torch.tensor(law_vectors)
    ).numpy().flatten()

    top_k_indices = similarities.argsort()[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]


    return top_k_articles

def openai_biencoder_retriever(openai_client, laws_df, chroma_collection, article_to_check, top_k=500, tokenizer = None):


    encoder = tokenizer

    # If no embeddings exist in the collection, encode the articles and store them
    if chroma_collection.count() == 0:

        ids = []
        documents = []
        embeddings = []

        idx = 0
        for _, row in tqdm(laws_df.iterrows()):
            article = row["contents"]

            if len(encoder.encode(article))>8191:
                # maxlength error
                continue

            # Call the OpenAI API to get embeddings for the article
            response = openai_client.embeddings.create(input=article, model=model_path)
            pooled_output = response.data[0].embedding

            ids.append(str(idx))
            documents.append(article)
            embeddings.append(pooled_output)
            idx += 1

        chunk_size = 1024
        total_chunks = len(embeddings) // chunk_size + 1

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


    # Retrieve conflicts for a specific article using either cosine similarity or classifier method
    top_conflicts = find_top_conflicts(article_to_check, openai_client, chroma_collection, top_k=top_k)

    return top_conflicts
