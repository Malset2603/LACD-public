#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=8  # Number of GPUs to use (0,1,2,3,4,5,6,7)

for seed in 0 1 2
do
    echo "=== Starting test for Seed ${seed} ==="
    
    # ReX
    gpu_id=$(( (seed*2 + 0) % NUM_GPUS ))
    echo "▶ Seed ${seed} - Running ReX... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/roberta-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "none" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --rex_conflict "train-generate" \
        --output_path "./outputs/retrieval_results/t-test/roberta-rex-trainGenerate-${seed}" &

    
    # GReX
    gpu_id=$(( (seed*2 + 1) % NUM_GPUS ))
    echo "▶ Seed ${seed} - Running GReX... (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
        --crossencoder_model_path "./data/models/LACD-cross/gnns/roberta-gat-${seed}" \
        --retrieval_method "re2" \
        --crossencoder_index_method "gat" \
        --rex_method "rex2" \
        --mode "test-benchmark" \
        --rex_conflict "train-generate" \
        --output_path "./outputs/retrieval_results/t-test/roberta-grex-trainGenerate-${seed}" &

    # Wait for all jobs in this fold to complete
done
wait

echo "All tests completed"
