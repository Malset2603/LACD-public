
# 모든 명령어를 백그라운드에서 동시에 실행
# TF-IDF
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method tfidf --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 100 --output_path ./outputs/retrieval_results/article_key/tfidf && echo "TF-IDF task completed!" &

# BM25
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method bm25 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 100 --output_path ./outputs/retrieval_results/article_key/bm25 && echo "BM25 task completed!" &

# enc-bi
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method retrieval --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 100 --output_path ./outputs/retrieval_results/article_key/enc-bi && echo "enc-bi task completed!" &

# Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/Re2 && echo "Re2 task completed!" &

# GAT
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-1 --retrieval_method re2 --crossencoder_index_method gat  --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/Re2+LGNN && echo "GAT task completed!" &

# ReX
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/ReX && echo "ReX task completed!" &

# GReX
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-1 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/GReX && echo "GReX task completed!" &

# 모든 백그라운드 작업이 완료될 때까지 대기
wait

python ./outputs/retrieval_results/article_key/data_key_refine.py

echo "All tasks have been completed successfully!"
