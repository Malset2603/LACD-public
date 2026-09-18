# Qwen reranker model
CUDA_VISIBLE_DEVICES=0,1 python ./src/encoders/cross_encoder/train/finetune-balanceWeight.py --mode train --tag qwen3-reranker-0 --method baseline --seed 0 --model "Qwen/Qwen3-Reranker-0.6B" --epoch 10 &

CUDA_VISIBLE_DEVICES=2,3 python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model "Qwen/Qwen3-Reranker-0.6B" --tag qwen3-reranker-gat-0 --gnn_method gat --epoch 10 --seed 0 --chroma_db_name kbb-baseline-nofinetune &


# Polyglot-ko reranker model
CUDA_VISIBLE_DEVICES=4,5 python ./src/encoders/cross_encoder/train/finetune-balanceWeight.py --mode train --tag hyperclovax-0 --method baseline --seed 0 --model "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B" --epoch 10 &

CUDA_VISIBLE_DEVICES=6,7 python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B" --tag hyperclovax-gat-0 --gnn_method gat --epoch 10 --seed 0 --chroma_db_name kbb-baseline-nofinetune &