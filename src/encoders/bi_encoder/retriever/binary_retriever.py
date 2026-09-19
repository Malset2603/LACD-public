import torch
from transformers import AutoTokenizer
import polars as pl
import chromadb

from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

# from src.utils.encoder.utils import MAX_TOKEN_LENGTH
import time

MAX_TOKEN_LENGTH = 4096
# MAX_TOKEN_LENGTH = 512

# our original code, bi-encoder.
def find_top_conflicts(article, model, tokenizer, chroma_collection, top_k=10, index_method = "none", max_length=None, fp16=False):
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
    # max_length caps sequence length (None = legacy module cap), always bounded by tokenizer.model_max_length
    ml = min(max_length or MAX_TOKEN_LENGTH, tokenizer.model_max_length)
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=ml,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    use_cuda = torch.cuda.is_available()
    with torch.inference_mode(), torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=bool(fp16) and use_cuda):
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


def binary_retriever(
    model,
    laws_df,
    chroma_collection,
    article_to_check,
    top_k=500,
    batch_size=8,
    tokenizer=None,
    allowed_keys=None,
    article_network=None,
    max_length=None,
    fp16=False,
    **kwargs
):
    """
    Bi-encoder retriever function to find conflicts in legal texts.

    Args:
        model: Pre-trained model module or str path to model directory.
        laws_df: Polars/Pandas DataFrame or str path to laws.csv.
        chroma_collection: Chroma DB collection object or str collection name.
        article_to_check: The article to check for conflicts.
        top_k: Number of top-k articles to retrieve.
        batch_size: Batch size for encoding if Chroma DB is empty.
        tokenizer: Tokenizer for the model.
        allowed_keys: Optional set of allowed article keys for filtering.
        article_network: Optional ArticleNetwork instance.
        max_length: Optional cap for tokenizer sequence length (None = legacy
            module cap MAX_TOKEN_LENGTH, always bounded by tokenizer.model_max_length).
        fp16: Enable fp16 autocast for encoding (CUDA only, no-op otherwise).

    Returns:
        tuple (pooled_output, top_k_articles)
    """
    # 1. Resolve model & tokenizer
    if isinstance(model, str):
        model_path = model
        loaded_model = torch.load(model_path + "/model.pth", weights_only=False)
        loaded_model.eval()
        if tokenizer is None:
            global _TOKENIZER_CACHE
            try:
                _TOKENIZER_CACHE
            except NameError:
                _TOKENIZER_CACHE = {}  # type: ignore
            if model_path in _TOKENIZER_CACHE:
                tokenizer = _TOKENIZER_CACHE[model_path]
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                _TOKENIZER_CACHE[model_path] = tokenizer
        model = loaded_model

    # 2. Resolve laws_df using Polars
    global _LAWS_DF_CACHE, _FILTERED_DF_CACHE, _SUBSET_FILTER_LOGGED
    try:
        _LAWS_DF_CACHE
    except NameError:
        _LAWS_DF_CACHE = {}  # type: ignore
        _FILTERED_DF_CACHE = {}  # type: ignore
        _SUBSET_FILTER_LOGGED = False  # type: ignore

    if isinstance(laws_df, str):
        laws_csv = laws_df
        if laws_csv not in _LAWS_DF_CACHE:
            _LAWS_DF_CACHE[laws_csv] = pl.read_csv(laws_csv, infer_schema_length=10000)
        laws_df = _LAWS_DF_CACHE[laws_csv]

    # 3. Filter laws_df by allowed_keys / article_network if provided
    if article_network is not None and hasattr(article_network, 'all_article_keys'):
        allowed_keys = set(article_network.all_article_keys)
    if allowed_keys is not None:
        cache_key = id(article_network) if article_network is not None else hash(frozenset(allowed_keys))
        if cache_key in _FILTERED_DF_CACHE:
            laws_df = _FILTERED_DF_CACHE[cache_key]
        else:
            before = laws_df.height
            filtered_df = laws_df.filter(
                pl.col("article_title").str.replace_all("·", "ㆍ").is_in(list(allowed_keys))
            )
            _FILTERED_DF_CACHE[cache_key] = filtered_df
            laws_df = filtered_df
            if not _SUBSET_FILTER_LOGGED:
                tqdm.write(f"[SUBSET] Chroma filter {before}->{laws_df.height} rows by allowed_keys ({len(allowed_keys)} keep)")
                _SUBSET_FILTER_LOGGED = True

    # 4. Resolve chroma_collection
    if isinstance(chroma_collection, str):
        from src.utils.encoder.biencoder_utils import load_chromaDB_byname
        chroma_collection = load_chromaDB_byname(chroma_collection)

    # 5. Populate Chroma DB if empty
    if chroma_collection.count() == 0:
        ids = []
        all_articles = laws_df["contents"].to_list()
        for idx in range(len(all_articles)):
            ids.append(str(idx))

        max_length = min(max_length or MAX_TOKEN_LENGTH, tokenizer.model_max_length) if tokenizer is not None else (max_length or MAX_TOKEN_LENGTH)
        use_cuda = torch.cuda.is_available()
        # fp16 autocast scope (CUDA only; no-op otherwise, never mutates the model)
        autocast_ctx = torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=bool(fp16) and use_cuda)

        embeddings = []
        documents = []

        # Adaptive batching with permanent step-down: on CUDA OOM, halve the
        # batch size globally (never scale back up) and re-split the failed
        # batch at the new size, so small-VRAM GPUs converge after ~1 OOM
        # instead of paying a synchronizing exception per batch.
        # OOM surfaces as torch.cuda.OutOfMemoryError (a RuntimeError subclass)
        # or plain RuntimeError on some builds, hence the message filter.
        enc_batch_size = max(1, batch_size)
        batch_start = 0
        pbar = tqdm(total=len(all_articles))
        while batch_start < len(all_articles):
            batch_articles = all_articles[batch_start:batch_start + enc_batch_size]
            batch_inputs = tokenizer(
                batch_articles,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = batch_inputs["input_ids"].to(model.encoder.device)
            attention_mask = batch_inputs["attention_mask"].to(model.encoder.device)

            try:
                with torch.inference_mode(), autocast_ctx:
                    encoder_to_use = model.passage_encoder if hasattr(model, "passage_encoder") else model.encoder
                    encoded_articles = encoder_to_use(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_outputs = encoded_articles.last_hidden_state[:, 0, :].cpu().numpy()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if enc_batch_size <= 1:
                    raise
                if use_cuda:
                    torch.cuda.empty_cache()
                enc_batch_size = max(1, enc_batch_size // 2)
                tqdm.write(f"[OOM] permanent step-down encode batch -> {enc_batch_size}, retrying (never scaling back up)")
                continue

            embeddings.extend(pooled_outputs)
            documents.extend(batch_articles)
            pbar.update(len(batch_articles))
            batch_start += len(batch_articles)
        pbar.close()

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

    # 6. Retrieve top conflicts
    article_vector, top_conflicts = find_top_conflicts(
        article_to_check, model, tokenizer, chroma_collection, top_k=top_k, max_length=max_length, fp16=fp16
    )

    return article_vector, top_conflicts


