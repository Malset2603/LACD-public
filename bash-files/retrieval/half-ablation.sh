#!/usr/bin/env bash
set -euo pipefail

# 고정 옵션
MODEL_PATH="./data/models/LACD-cross/gnns/roberta-gat-0"
RETRIEVAL_METHOD="re2"
INDEX_METHOD="gat"
REX_METHOD="rex2"
MODE="test-benchmark"

# rex_kI 값 배열
REX_KI_VALUES=(40 30 20 10)
OUTPUT_DIRS=(
  "./outputs/retrieval_results/roberta-gat-0-rex2-40pmedian"
  "./outputs/retrieval_results/roberta-gat-0-rex2-30pmedian"
  "./outputs/retrieval_results/roberta-gat-0-rex2-20pmedian"
  "./outputs/retrieval_results/roberta-gat-0-rex2-10pmedian"
)

NUM_GPUS=5  # 사용할 GPU 수 (필요에 따라 조정)

# 각 rex_kI 값을 GPU에 round-robin 방식으로 할당
for i in "${!REX_KI_VALUES[@]}"; do
  rex_kI=${REX_KI_VALUES[$i]}
  output_dir=${OUTPUT_DIRS[$i]}
  gpu_id=$(( i % NUM_GPUS ))
  
  echo "=== rex_kI = ${rex_kI} 테스트 시작 (GPU ${gpu_id}) ==="
  echo "▶ [GPU ${gpu_id}] rex_kI=${rex_kI} → ${output_dir}"
  
  CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
    --crossencoder_model_path "$MODEL_PATH" \
    --retrieval_method "$RETRIEVAL_METHOD" \
    --crossencoder_index_method "$INDEX_METHOD" \
    --rex_method "$REX_METHOD" \
    --mode "$MODE" \
    --rex_kI ${rex_kI} \
    --output_path "$output_dir" &
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "모든 테스트 완료"