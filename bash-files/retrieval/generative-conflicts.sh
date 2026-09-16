
# Generate C using Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 100 --output_path ./outputs/retrieval_results/generative-conflicts/re2-0 --train_query_path ./data/datasets/LACD-retrieval/train-queries.jsonl

# Generate C using GAT
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat --mode test-benchmark --biencoder_top_k 100 --output_path ./outputs/retrieval_results/generative-conflicts/re2-gat-0 --train_query_path ./data/datasets/LACD-retrieval/train-queries.jsonl

# ReX using synthetic C
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/generative-conflicts/re2-rex-train-generate-0 --rex_conflict train-generate

# GReX using synthetic C
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/generative-conflicts/grex-train-generate-0 --rex_conflict train-generate

