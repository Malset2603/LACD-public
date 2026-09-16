

for seed in {0..2}
do 
# 각 명령어를 백그라운드에서 실행하고 GPU 할당
python ./src/encoders/cross_encoder/train/finetune-balanceWeight.py --mode train --tag roberta-${seed} --method baseline --seed ${seed} --model klue/roberta-base --epoch 10 

python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model klue/roberta-base --tag roberta-gat-${seed} --gnn_method gat --epoch 10 --seed ${seed} --chroma_db_name kbb-baseline-nofinetune 


python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model klue/roberta-base --tag roberta-gcn-${seed} --gnn_method gcn --epoch 10 --seed ${seed} --chroma_db_name kbb-baseline-nofinetune 



python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model klue/roberta-base --tag roberta-graphsage-${seed} --gnn_method graphsage --epoch 10 --seed ${seed} --chroma_db_name kbb-baseline-nofinetune 

python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model klue/roberta-base --tag roberta-vanilla-${seed} --gnn_method vanilla --epoch 10 --seed ${seed} --chroma_db_name kbb-baseline-nofinetune 

done