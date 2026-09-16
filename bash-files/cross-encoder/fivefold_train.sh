#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for fold_idx in 0 1 2 3 4
do
    # baseline 모델 학습
    echo "Fold ${fold_idx} baseline 모델 학습 시작..."
    python ./src/encoders/cross_encoder/train/finetune-balanceWeight.py \
        --mode train \
        --tag roberta-baseline-42-fold${fold_idx} \
        --method baseline \
        --seed 42 \
        --model klue/roberta-base \
        --epoch 10 \
        --fivefold True \
        --fivefold_num ${fold_idx}
    
    echo "Fold ${fold_idx} baseline 모델 학습 완료"

    # GAT 모델 학습
    echo "Fold ${fold_idx} GAT 모델 학습 시작..."
    python ./src/methods/LawGNN/train/crossencoder_finetune.py \
        --mode train \
        --model klue/roberta-base \
        --tag roberta-gat-42-fold${fold_idx} \
        --gnn_method gat \
        --epoch 10 \
        --seed 42 \
        --chroma_db_name kbb-baseline-nofinetune \
        --fivefold True \
        --fivefold_num ${fold_idx}
    
    echo "Fold ${fold_idx} GAT 모델 학습 완료"
done

echo "모든 fold 학습 완료"
