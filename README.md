# Legal Article Competition Detection (LACD)

An official repository for *A Method for Detecting Legal Article Competition for Korean Criminal Law Using a Case-augmented Mention Graph* paper.

![github-image.v1.svg](./figs/github-image.v2.svg)

## Installation

### **Virtual environment**

```bash
conda create -n lacd python=3.9
conda activate lacd
pip install -r requirements.txt
```

Before you start, make sure your default python execution path is our base directory. You can change it by `export PYTHONPATH=.`

<!-- ### **Chroma DB**

```bash
pip install chromadb
pip install chromadb-client
``` -->

### **Law download**

Please locate provided `/data/database` folder in `/data/database`.
You can download it in [here](https://drive.google.com/file/d/1TcXfcGUlww9w6quWaSon3RYxqt_Q1Od9/view?usp=sharing).

### **Setting SEED value**

You can change the seed value by changing following codes in `/src/utils/utils.py` file.

```python
SEED = 42

def article_key_function(text)->str:
    import re
    #...
```

## Train

### Bi-encoder train

```bash
# Naïve Re2 bi-encoder
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-baseline

# Ours
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-caseaug --method case-augmentation

# Debug - 20 samples, 1 epoch, batch 1, 512 tokens, fp16 (fast smoke test, 2GB GPU)
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-baseline-debug --debug --debug_limit 20 --batch_size 1 --max_length 512 --fp16 --gradient_checkpointing
```

The models are saved in `/data/models/LACD-bi`. 
<!-- Alternatively, you can use provided model checkpoints. -->

### Build Chroma DB from scratch

By executing following codes, Chroma DBs are automatically built.

```bash
# Naïve Re2 bi-encoder chroma DB (kbb-baseline)
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bi-only

# Ours bi-encoder chroma DB (kbb-caseaug)
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug --chroma_db_name kbb-caseaug --biencoder_method caseaug --retrieval_method bi-only

# Debug (inference single article, no benchmark loop - already fast)
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bi-only --biencoder_top_k 2
```

Chroma DBs are saved in `/data/database/chroma_db`. Note that it takes amount of time. Alternatively, you can use provided Chroma DB file.

### Cross encoder train (Experiments for Step 3)

```bash
# Naïve Re2 cross-encoder
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bi-only

# Ours
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gat-caseaugembds --gnn_method gat --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3
```

Without `--model` option, our code trains `monologg/kobigbird-bert-base` (KoBigBird) model. After training, the models are saved in `/data/models/LACD-cross`. 
<!-- Alternatively, you can use provided model checkpoints -->

Alternatively, you can use the following code.
```bash
bash bash-files/cross-encoder/cross_train_seedset.sh
```


## Inference

### Retrieve competing articles (Experiments for all Steps)

You can check the retrieval result by following code

```bash
# Naïve Re2 retriever
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline  --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline

# CAM-Re2
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/kbb-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug  --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method caseaug

# Debug - single article + small top_k (fast, no model change)
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline --biencoder_top_k 2 --crossencoder_top_k 2

# Debug - benchmark on 5 samples only, output -> outputs/retrieval_results/*_debug.jsonl
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline --mode test-benchmark --debug --debug_limit 5

# Custom output directory to avoid overwriting across experiments
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline --mode test-benchmark --output_dir ./outputs/my_experiment --output_name my_run

# Debug - classical retrievers (no GPU, fastest)
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method tfidf --biencoder_top_k 2
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bm25 --biencoder_top_k 2
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method tfidf --mode test-benchmark --debug --debug_limit 5
```

### Evaluating retrieval results

`--mode test-benchmark` only saves `outputs/retrieval_results/*.jsonl` (no metrics printed). Evaluate separately:

```bash
# Single file (subset example)
python src/eval/retrieval_eval.py --result_path ./outputs/retrieval_results/baseline_gat_baseline_laws2000.jsonl --top_k 5

# Whole directory (all experiments in custom output_dir)
python src/eval/retrieval_eval.py --result_path ./outputs/my_experiment --top_k 5 --ground_truth_path ./data/datasets/LACD-biclassification/train-test-divide/test.jsonl

# Save metrics to file (optional, no default — if not provided, only printed)
python src/eval/retrieval_eval.py --result_path ./outputs/my_experiment --top_k 5 --metrics_output ./outputs/my_experiment/metrics.json
```

### Saving metrics (optional, all scripts)

All training and evaluation scripts support `--metrics_output` with **no default**. If not provided, metrics are only printed. If provided, the directory is auto-created and the file is overwritten:

```bash
# Bi-encoder training
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-baseline --metrics_output ./outputs/metrics/bi-baseline.json

# Cross-encoder GNN training
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-gat --gnn_method gat --chroma_db_name kbb-baseline --metrics_output ./outputs/metrics/cross-gat.json
python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-gat-noLM --gnn_method gat --chroma_db_name kbb-baseline --metrics_output ./outputs/metrics/cross-gat-noLM.json

# Retrieval evaluation
python src/eval/retrieval_eval.py --result_path ./outputs/my_experiment --metrics_output ./outputs/my_experiment/metrics.json
```

### Testing query processing time

You can check query processing time (One of Appendix experiments) by following code.

```bash
bash ./bash-files/retrieval/qps-test.sh
# Debug - single top_k
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/kbb-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline --biencoder_top_k 2
```

## Ablation studies


### Step 3 experiment without cross encoder

You can check CAM-Re2 without cross encoder by following code.
```bash
bash bash-files/cross-encoder/noLM-crossencoder_train.sh
```


### GNN architectures comparison

You can check alternative GNN architectures (GCN and GraphSAGE) for CAM-Re2 by following code.

```bash
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gcn-caseaugembds --gnn_method gcn --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3


python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-graphsage-caseaugembds --gnn_method graphsage --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3
```

### Subset sampling via arguments (fast yet representative — for thesis experiments)

> A **stratified**, **graph-aware** alternative to `--debug`. No need to create a new dataset, simply add arguments. Early-load 2-pass streaming (never loads 100% into RAM), deterministic via `--subset_seed` — `pass1` only counts `label/degree` (`indices`/`Counter`), `pass2` only loads `keep` (`Random(seed+label)` per class + `Random(seed)` final shuffle).

| Argument | Default | Effect |
|----------|---------|--------|
| `--subset_ratio 0.15` | `None` (full) | Early-load stratified fraction `(0,1]` for `train/val/test.jsonl` preserving `P(y=1)=12.8%` via `src/utils/utils.py:28`. E.g., `0.15` = `1399->209`, `469->70`. Statistically representative, unlike `--debug` `head(N)`. Deterministic across sessions. Alias: `--sample_ratio`, `--mini_ratio` (deprecated). |
| `--subset_laws 2000` | `None` (79k) | Early-load `LMGraph` `ArticleNetwork` `src/methods/LawGNN/article_network/article_network.py:21` — degree is computed first from `law_link`, then `laws.csv` is streamed keeping only `keep` with hub-preserving `50%` top-degree + `50%` random. `2000` nodes `~7.9k` edges vs `79k/339k` full. Chroma build `~4 min` vs `~60 min`. Alias: `--mini_laws`, `--law_nodes` (deprecated). |
| `--subset_seed 42` | `42` | Reproducible seed for both samplings above (`seed+label` per class). Alias: `--mini_seed` (deprecated). |
| `--output_dir ./outputs/retrieval_results` | `./outputs/retrieval_results` | Custom directory for `test-benchmark` results to avoid overwriting across experiments, e.g. `--output_dir ./outputs/my_experiment`. Auto-created. |
| `--output_name my_run` | `None` (auto) | Custom filename (without `.jsonl`) for retrieval results; if not set, auto-generated as `{biencoder_method}_{crossencoder_index_method}_{crossencoder_method}{suffix}.jsonl`. |

Subset outputs are automatically suffixed `_subset15_laws2000` to avoid overwriting full results, e.g., `*_subset15_laws2000.jsonl` (legacy `_mini*` still readable via alias).

**When to use which:**
* `debug` (`head`): bug smoke test, `<10s`, metrics **not** representative.
* `subset` (`stratified`): hyperparameter tuning (`tau/alpha/epoch`), GNN ablations, `Spearman Recall@50 >0.85` vs full, `~10x` faster.

```bash
# Bi-encoder training — subset 15% + 2k articles, 512 tokens, fp16 (4 min on 2GB GPU) — recommended default for thesis
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-mini15 --subset_ratio 0.15 --subset_laws 2000 --max_length 512 --fp16 --batch_size 4 --epoch 3

# Bi-encoder training — subset 10% for rapid sweeps
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-mini10 --subset_ratio 0.1 --subset_laws 2000 --max_length 512 --fp16 --batch_size 4 --epoch 1

# Build Chroma DB subset (2k articles only) — must use the same subset_laws as training to keep the graph consistent
python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-mini15 --chroma_db_name kbb-mini --biencoder_method baseline --retrieval_method bi-only --subset_laws 2000

# Cross-encoder GNN training — subset 10%
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-gat-mini10 --gnn_method gat --chroma_db_name kbb-mini --case_augmentation_method baseline --epoch 3 --subset_ratio 0.1 --subset_laws 2000 --max_length 512 --fp16

# Benchmark — subset 20% (93 samples instead of 469) — 5x faster, still stratified
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/kbb-gat-mini10 --biencoder_model_path ./data/models/LACD-bi/kbb-mini15 --chroma_db_name kbb-mini --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method baseline --mode test-benchmark --subset_ratio 0.2 --subset_laws 2000
# output -> outputs/retrieval_results/*_subset20_laws2000.jsonl

# Classical retriever subset (no GPU, <10s)
python ./src/main.py --chroma_db_name kbb-mini --retrieval_method tfidf --subset_laws 2000 --subset_ratio 0.2 --mode test-benchmark

# Combine subset + debug for ultra-fast smoke test (5 samples drawn from the 15% stratified subset)
python ./src/main.py --chroma_db_name kbb-mini --retrieval_method hybrid --subset_ratio 0.15 --subset_laws 2000 --mode test-benchmark --debug --debug_limit 5 --biencoder_top_k 2 --crossencoder_top_k 2
```

> Tip: `subset` and `debug` can be combined. `subset_ratio` is applied first (stratified `209`), then `debug` takes `head(5)` from that subset. `mini_*` aliases remain functional but deprecated.

### Debug mode

All `src/main.py` runs support `--debug --debug_limit N` (default 5). In `test-benchmark` mode it slices `data/datasets/LACD-biclassification/train-test-divide/test.jsonl` to `N` samples and writes `*_debug.jsonl` to avoid overwriting full results. In `inference` mode use `--biencoder_top_k 2 --crossencoder_top_k 2` or `tfidf`/`bm25` for the fastest check. `src/encoders/bi_encoder/train/finetune.py` also supports `--debug --debug_limit 20 --batch_size 1 --max_length 512 --fp16 --gradient_checkpointing` (slices train/val/test to 20 samples, forces epoch=1; defaults `batch_size 4`, `max_length 4096`, `fp16/gradient_checkpointing false` — use small values only for 2GB VRAM debugging).

```bash
# Example: every README command has a debug counterpart
# Full:
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/kbb-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method caseaug --mode test-benchmark
# Debug (5 samples):
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/kbb-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method caseaug --mode test-benchmark --debug --debug_limit 5
```
