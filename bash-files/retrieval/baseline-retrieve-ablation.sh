#!/usr/bin/env bash
set -euo pipefail

# 실험할 파라미터
# THETAS=(0.001 0.01 0.1 0.5 0.9 0.99 0.999)
# THETAS=(0)
# KB_VALUES=(79615)

# TOP_K=(134 117 114 112 111 109 100)
TOP_K=(150)

BASE_OUTPUT="./outputs/retrieval_results/theta-kB/"

# 고정 옵션
MODEL_PATH="./data/models/LACD-cross/roberta-0"
RETRIEVAL_METHOD="re2"
INDEX_METHOD="none"
MODE="test-benchmark"

NUM_GPUS=7  # 사용할 GPU 수 (0,1,2,3,4,5,6)

# 각 theta 값을 GPU에 round-robin 방식으로 할당
for i in "${!TOP_K[@]}"; do
  top_k=${TOP_K[$i]}
  top_k_fmt=$(printf "%d" "$top_k")
  gpu_id=$(( i % NUM_GPUS ))
  
  echo "=== k = ${top_k} 테스트 시작 (GPU ${gpu_id}) ==="

  # 각 kB 값에 대해 동일한 GPU 사용
  out_dir="${BASE_OUTPUT}/baseline-top_k_${top_k_fmt}"

  echo "▶ [GPU ${gpu_id}] top_k=${top_k_fmt} → ${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
    --crossencoder_model_path "$MODEL_PATH" \
    --retrieval_method "$RETRIEVAL_METHOD" \
    --crossencoder_index_method "$INDEX_METHOD" \
    --mode "$MODE" \
    --output_path "$out_dir" \
    --biencoder_top_k "$top_k_fmt" &
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "모든 테스트 완료"