


# Hyperclovax
# Re2
CUDA_VISIBLE_DEVICES=0 python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/hyperclovax-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/hyperclovax-re2-0 &

# GReX
CUDA_VISIBLE_DEVICES=1 python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/hyperclovax-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/hyperclovax-grex-0 &


# OpenAI biencoder
CUDA_VISIBLE_DEVICES=2 python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_model_path text-embedding-3-small --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/re2-embedding3-0 --chroma_db_name openai-3small


# OpenAI GReX
CUDA_VISIBLE_DEVICES=3 python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --biencoder_model_path text-embedding-3-small --output_path ./outputs/retrieval_results/article_key/grex-embedding3-0 --chroma_db_name openai-3small
wait