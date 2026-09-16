
# for several different thresholds

# GPU별로 parallel하게 실행하기 위한 배열
thresholds=(0.01 0.1 0.9 0.99)
gpu_count=$(nvidia-smi -L | wc -l)

# # Generate C using GAT - parallel execution
# for i in "${!thresholds[@]}"
# do
#     threshold=${thresholds[$i]}
#     gpu_id=$((i % gpu_count))
    
#     CUDA_VISIBLE_DEVICES=$gpu_id python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat --mode train-generate --biencoder_top_k 100 --output_path ./outputs/retrieval_results/generative-conflicts/re2-gat${threshold}-0 --train_query_path ./data/datasets/LACD-retrieval/train-queries.jsonl --train_generate_threshold $threshold &
# done

# 모든 background 프로세스가 완료될 때까지 대기
# wait



# GReX using synthetic C - parallel execution
for i in "${!thresholds[@]}"
do
    threshold=${thresholds[$i]}
    gpu_id=$((i % gpu_count))
    
    CUDA_VISIBLE_DEVICES=$gpu_id python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/generative-conflicts/grex-train-generate-${threshold}-0 --rex_conflict train-generate --train_generate_threshold $threshold &
done

# 모든 background 프로세스가 완료될 때까지 대기
wait