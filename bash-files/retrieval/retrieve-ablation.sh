#!/usr/bin/env bash
set -euo pipefail

# 실험할 파라미터
THETAS=(0 0.001 0.01 0.1 0.5 0.9 0.99 0.999)
KB_VALUES=(79615)

BASE_OUTPUT="./outputs/retrieval_results/theta-kB/"

# 고정 옵션
MODEL_PATH="./data/models/LACD-cross/gnns/roberta-gat-0"
RETRIEVAL_METHOD="re2"
INDEX_METHOD="gat"
REX_METHOD="rex2"
MODE="test-benchmark"

NUM_GPUS=7  # 사용할 GPU 수 (0,1,2,3,4,5,6)

# 각 theta 값을 GPU에 round-robin 방식으로 할당
for i in "${!THETAS[@]}"; do
  theta=${THETAS[$i]}
  theta_fmt=$(printf "%.4f" "$theta")
  gpu_id=$(( i % NUM_GPUS ))
  
  echo "=== θ = ${theta_fmt} 테스트 시작 (GPU ${gpu_id}) ==="

  # 각 kB 값에 대해 동일한 GPU 사용
  for kB in "${KB_VALUES[@]}"; do
    kB_int=$(printf "%d" "$kB")
    out_dir="${BASE_OUTPUT}/theta_${theta_fmt}_kB_${kB_int}-test-generate"

    echo "▶ [GPU ${gpu_id}] θ=${theta_fmt}, kB=${kB_int} → ${out_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python ./src/main.py \
      --crossencoder_model_path "$MODEL_PATH" \
      --retrieval_method "$RETRIEVAL_METHOD" \
      --crossencoder_index_method "$INDEX_METHOD" \
      --rex_method "$REX_METHOD" \
      --mode "$MODE" \
      --output_path "$out_dir" \
      --rex_kB "$kB_int" \
      --rex_filtering_threshold "$theta_fmt" \
      --rex_conflict "test-generate" &
  done
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "모든 테스트 완료"