from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
from src.utils.encoder.biencoder_utils import load_chromaDB_byname

# Module-level caches keyed by chroma_db_name. The law corpus is static within
# a run, so the collection handle, document list, and fitted TF-IDF/BM25 index
# are built once; per-query work is only transform/score. Keying by db name
# prevents cross-contamination between experiments sharing one process.
_COLLECTION_CACHE = {}
_DOCS_CACHE = {}
_TFIDF_CACHE = {}
_BM25_CACHE = {}


def _get_collection(chroma_db_name):
    if chroma_db_name not in _COLLECTION_CACHE:
        _COLLECTION_CACHE[chroma_db_name] = load_chromaDB_byname(chroma_db_name)
    return _COLLECTION_CACHE[chroma_db_name]


def _get_law_documents(chroma_db_name):
    # Fetch once; embeddings are never used by classical retrievers.
    if chroma_db_name not in _DOCS_CACHE:
        collection = _get_collection(chroma_db_name)
        _DOCS_CACHE[chroma_db_name] = collection.get(include=["documents"])["documents"]
    return _DOCS_CACHE[chroma_db_name]

def find_top_conflicts_with_tfidf(article, chroma_db_name, top_k=10):
    """
    Function to find top conflicts using TF-IDF retriever.
    
    Args:
    article (str): The input article to check.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_db_name: Chroma DB name containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.

    NOTE: the TF-IDF index is fit on the law corpus only (standard practice);
    the query is transformed, not fit. The old code fit on corpus + query,
    which leaked the query into IDF values. Rankings are effectively the
    same; verify parity on retrieval metrics before paper runs.
    """

    law_documents = _get_law_documents(chroma_db_name)

    # Fit once per DB; per query only transform.
    if chroma_db_name not in _TFIDF_CACHE:
        tfidf_vectorizer = TfidfVectorizer()
        law_vectors = tfidf_vectorizer.fit_transform(law_documents)
        _TFIDF_CACHE[chroma_db_name] = (tfidf_vectorizer, law_vectors)
    else:
        tfidf_vectorizer, law_vectors = _TFIDF_CACHE[chroma_db_name]

    article_vector = tfidf_vectorizer.transform([article])

    similarities = np.dot(law_vectors, article_vector.T).toarray().flatten()
    top_k_indices = similarities.argsort()[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    return top_k_articles


def find_top_conflicts_with_bm25(article, chroma_db_name, top_k=10):
    """
    Function to find top conflicts using BM25 retriever.
    
    Args:
    article (str): The input article to check.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_db_name: Chroma DB name containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.

    The BM25 index is fit on the law corpus only (as before) and cached per
    DB; per query only tokenizes and scores. Bit-identical to the old code.
    """

    law_documents = _get_law_documents(chroma_db_name)

    # Fit once per DB; per query only tokenize and score.
    if chroma_db_name not in _BM25_CACHE:
        corpus = [doc.split() for doc in law_documents]  # Split documents into words
        _BM25_CACHE[chroma_db_name] = BM25Okapi(corpus)
    bm25 = _BM25_CACHE[chroma_db_name]

    # Tokenize the input article
    article_tokens = article.split()

    # Get BM25 scores
    scores = bm25.get_scores(article_tokens)
    
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    return top_k_articles
