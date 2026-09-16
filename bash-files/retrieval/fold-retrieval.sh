#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=7  # 사용할 GPU 수 (0,1,2,3,4,5,6)

for fold_idx in 0 1 2 3 4
do
    echo "=== Fold ${fold_idx} 테스트 시작 ==="
    
    # Re2
    gpu_id=$(( fold_idx % NUM_GPUS ))
    echo "▶ Fold ${fold_idx} - Re2 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/roberta-baseline-42-fold${fold_idx}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "none" \
        --mode "test-benchmark" \
        --biencoder_top_k "142" \
        --multi-fold True \
        --multi-fold-k "${fold_idx}" \
        --output_path "./outputs/retrieval_results/multi-fold/fold_${fold_idx}/re2-142" &
    
    # GAT 
    gpu_id=$(( (fold_idx + 1) % NUM_GPUS ))
    echo "▶ Fold ${fold_idx} - GAT 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/gnns/roberta-gat-42-fold${fold_idx}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "gat" \
        --mode "test-benchmark" \
        --biencoder_top_k "142" \
        --multi-fold True \
        --multi-fold-k "${fold_idx}" \
        --output_path "./outputs/retrieval_results/multi-fold/fold_${fold_idx}/gat-142" &
    
    # ReX
    gpu_id=$(( (fold_idx + 2) % NUM_GPUS ))
    echo "▶ Fold ${fold_idx} - ReX 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/roberta-0" \
        --retrieval_method "re2" \
        --crossencoder_index_method "none" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --multi-fold True \
        --multi-fold-k "${fold_idx}" \
        --output_path "./outputs/retrieval_results/multi-fold/fold_${fold_idx}/rex-median-theta" &
    
    # GReX
    gpu_id=$(( (fold_idx + 3) % NUM_GPUS ))
    echo "▶ Fold ${fold_idx} - GReX 실행 중... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/gnns/roberta-gat-0" \
        --retrieval_method "re2" \
        --crossencoder_index_method "gat" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --multi-fold True \
        --multi-fold-k "${fold_idx}" \
        --output_path "./outputs/retrieval_results/multi-fold/fold_${fold_idx}/grex-median-theta" &

    # 각 fold의 모든 작업이 끝날 때까지 대기
    wait
done

echo "모든 테스트 완료"