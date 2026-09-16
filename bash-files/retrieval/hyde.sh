# roberta hyde
python ./src/main.py --retrieval_method hyde --crossencoder_index_method none  --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/roberta-hyde --biencoder_top_k 150

# openai hyde
python ./src/main.py --retrieval_method hyde --biencoder_model_path text-embedding-3-small --chroma_db_name openai-3small --crossencoder_index_method none  --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/openai-3small-hyde --biencoder_top_k 150
