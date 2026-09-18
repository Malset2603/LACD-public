# seed 0
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_re2-0 --biencoder_top_k 150

python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_gat-0 --biencoder_top_k 150


# seed 1
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-1 --retrieval_method re2 --crossencoder_index_method none  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_re2-1 --biencoder_top_k 150

python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-1 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_gat-1 --biencoder_top_k 150



# seed 2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-2 --retrieval_method re2 --crossencoder_index_method none  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_re2-2 --biencoder_top_k 150

python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-2 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_gat-2 --biencoder_top_k 150