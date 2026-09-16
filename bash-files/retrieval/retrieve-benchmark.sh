# Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/re2-0

# Re2 + LGNN
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/re2-gat-0

# ReX + Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/re2-rex-0

# GReX
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/grex-0
 