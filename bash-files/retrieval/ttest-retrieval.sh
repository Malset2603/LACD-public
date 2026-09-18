#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=8  # 사용할 GPU 수 (0,1,2,3,4,5,6,7)

for seed in 0 1 2
do
    echo "=== Seed ${seed} 테스트 시작 ==="
    
    # Re2
    gpu_id=$(( seed % NUM_GPUS ))
    echo "▶ Seed ${seed} - Re2 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/roberta-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "none" \
        --mode "test-benchmark" \
        --biencoder_top_k 142 \
        --output_path "./outputs/retrieval_results/t-test/roberta-re2-${seed}" &
    
    # GAT   
    gpu_id=$(( (seed + 1) % NUM_GPUS ))
    echo "▶ Seed ${seed} - GAT 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/gnns/roberta-gat-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "gat" \
        --mode "test-benchmark" \
        --biencoder_top_k 142 \
        --output_path "./outputs/retrieval_results/t-test/roberta-gat-${seed}" &
    
    # ReX
    gpu_id=$(( (seed + 2) % NUM_GPUS ))
    echo "▶ Seed ${seed} - ReX 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/roberta-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "none" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --output_path "./outputs/retrieval_results/t-test/roberta-rex-${seed}" &
    
    # GReX
    gpu_id=$(( (seed + 3) % NUM_GPUS ))
    echo "▶ Seed ${seed} - GReX 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/gnns/roberta-gat-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "gat" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --output_path "./outputs/retrieval_results/t-test/roberta-grex-${seed}" &

    # 각 fold의 모든 작업이 끝날 때까지 대기
    wait
done

echo "모든 테스트 완료"