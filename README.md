# Legal Article Conflict Detection (LACD)

An official repository for *GReX: A Graph Neural Network-based Rerank-then-Expand Method for Detecting Legal Article Conflict in Korean Criminal Law* paper.

![github-image.v1.svg](./figs/main-image.svg)

## Installation

### **Virtual environment**

```bash
conda create -n lacd python=3.9
conda activate lacd
pip install -r requirements.txt
```

Before you start, make sure your default python execution path is our base directory. To do this, you should execute `export PYTHONPATH=.` in each terminal.


### **Law download**

Please locate provided law corpus data in `/data`.

### **Setting SEED value (optional)**

You can change the seed value by changing following codes in `/src/utils/utils.py` file.

```python
SEED = 42

def article_key_function(text)->str:
    import re
    #...
```

## Train

### Bi-encoder prepare

```bash
# Pretrained model
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --epoch 0 --tag kbb-baseline-nofinetune
```

The model will be saved in `/data/models/LACD-bi/kbb-baseline-nofinetune`. 
<!-- Alternatively, you can use provided model checkpoints. -->

### Build Chroma DB from scratch

By executing following codes, Chroma DBs are automatically built.

```bash
# Naïve Re2 bi-encoder chroma DB (kbb-baseline)
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline-nofinetune --chroma_db_name kbb-baseline-nofinetune --retrieval_method retrieval
```

Chroma DBs are saved in `/data/database/chroma_db`. Note that it takes amount of time (around ~20 minutes for eight Nvidia TITAN RTX GPUs).

### Cross encoder train

```bash
# baseline (Re2) cross-encoder
python ./src/encoders/cross_encoder/train/finetune-balanceWeight.py --mode train --tag roberta-0 --method baseline --model klue/roberta-base --epoch 10 

# Our LGNN cross-encoder
python ./src/methods/LawGNN/train/crossencoder_finetune.py --mode train --model klue/roberta-base --tag roberta-gat-0 --gnn_method gat --epoch 10 --chroma_db_name kbb-baseline-nofinetune 
```

Without `--model` option, our code trains `klue/roberta-base` model. After training, the models are saved in `/data/models/LACD-cross`.
These codes are all provided in `/bash-files/cross-encoder/all-train.sh`. 
<!-- Alternatively, you can use provided model checkpoints -->

## Test

### Main Results: retrieve competing articles in the LACD dataset

You can conduct main experiments by using following codes

```bash
# Re2 retriever
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode test-benchmark --biencoder_top_k 150 --output_path ./outputs/retrieval_results/article_key/re2-0.jsonl

# GReX retriever
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/grex-0.jsonl
```

Alternatively, you can execute `/bash-files/retrieval/retrieve-benchmark.sh`

You can evaluate results by following codes
```bash
# article post-processing
python ./src/eval/data-key-refine.py --input_dir ./outputs/retrieval_results/article_key --output_dir ./outputs/retrieval_results/article_key

python ./src/eval/retrieval_eval.py
```

### ReX by synthetic C

To evaluate ReX using synthetic C, use following codes:

```bash
# Generate C using Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none --mode train-generate --biencoder_top_k 100 --output_path ./outputs/retrieval_results/generative-conflicts/re2-0

# Generate C using GAT
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat --mode train-generate --biencoder_top_k 100 --output_path ./outputs/retrieval_results/generative-conflicts/re2-gat-0

# ReX using synthetic C
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/re2-rex-train-generate-0 --rex_conflict train-generate

# GReX using synthetic C
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rex2 --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/grex-train-generate-0 --rex_conflict train-generate
```

### Other baselines in LACD

You can conduct Rocchio-PRF experiments by following codes
```bash
(lacd) /LACD$ python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/roberta-0 --retrieval_method re2 --crossencoder_index_method none  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_re2-0 --biencoder_top_k 150

(lacd) /LACD$ python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/roberta-gat-0 --retrieval_method re2 --crossencoder_index_method gat  --rex_method rocchio --mode test-benchmark --output_path ./outputs/retrieval_results/article_key/rocchioprf_gat-0 --biencoder_top_k 150
```

You can evaluate results using the code in the main experiment.
