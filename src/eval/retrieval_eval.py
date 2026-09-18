import sys
import os

# Ensure repository root is on sys.path when invoked directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import glob
import json
from tqdm import tqdm
from src.utils.utils import article_key_function

# ANSI escape codes for table formatting
BLUE_BOLD = '\033[1;34m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'

# ---------------------------------------------------------------------------
# Standard / Pipeline Evaluation (Precision, Recall, F1, Missed Pairs)
# ---------------------------------------------------------------------------
def load_ground_truth(ground_truth_path):
    """Load ground truth from test.jsonl (current layout) with fallback to legacy paths."""
    if ground_truth_path and os.path.exists(ground_truth_path):
        paths = [ground_truth_path]
    else:
        # fallback candidates for backward compatibility
        candidates = [
            ground_truth_path,
            "./data/datasets/LACD-biclassification/train-test-divide/test.jsonl",
            "./data/datasets/LACD-retrieval/test.jsonl",
            "./data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl",
        ]
        paths = [p for p in candidates if p and os.path.exists(p)]
        if not paths:
            raise FileNotFoundError(f"Ground truth not found. Tried: {candidates}. Use --ground_truth_path to specify correct path.")
        if len(paths) > 1:
            print(f"[WARN] multiple ground truth candidates found, using {paths[0]}")
        paths = paths[:1]

    rows = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    # also include missed_pairs if exists (legacy)
    legacy_missed = "./data/datasets/LACD-retrieval/missed_pairs.jsonl"
    if os.path.exists(legacy_missed) and legacy_missed not in paths:
        with open(legacy_missed, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except:
                    pass
    # normalize
    normalized = []
    for r in rows:
        try:
            normalized.append({
                "article1": article_key_function(r["article1"]),
                "article2": article_key_function(r["article2"]),
                "answer": r["answer"]
            })
        except:
            continue
    return normalized


def evaluate_single(result_path, rows, top_k=5, biencoder_top_k=10):
    print(f"==\nresult: {result_path}")
    results = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    checked_articles = set()
    true_positive = 0
    false_positive = 0
    blind_negative = 0
    false_negative = 0
    blind_positive = 0
    missed_pairs = []
    seen_pairs = set()

    for result in results:
        article_to_check = result['article_to_check']
        if article_to_check in checked_articles:
            continue
        checked_articles.add(article_to_check)
        articles = result['articles']
        articles = [a for a in articles if article_key_function(article_to_check) != article_key_function(a)]

        if len(articles) == 0:
            matching_rows = [r for r in rows if article_key_function(r["article1"]) == article_key_function(article_to_check) or article_key_function(r["article2"]) == article_key_function(article_to_check)]
            has_true = any(r['answer'] is True for r in matching_rows)
            has_false = any(r['answer'] is False for r in matching_rows)
            if has_true:
                false_negative += 1
            elif has_false:
                blind_negative += 1
        else:
            for a in articles[:top_k]:
                matching_rows = [r for r in rows if (article_key_function(r["article1"]) == article_key_function(article_to_check) and article_key_function(r["article2"]) == article_key_function(a)) or (article_key_function(r["article2"]) == article_key_function(article_to_check) and article_key_function(r["article1"]) == article_key_function(a))]
                if len(matching_rows) == 0:
                    pair = (article_key_function(article_to_check), article_key_function(a))
                    reverse_pair = (pair[1], pair[0])
                    if pair not in seen_pairs and reverse_pair not in seen_pairs:
                        missed_pairs.append({"article1": article_to_check, "article2": a, "answer": None})
                        seen_pairs.add(pair)
                        blind_positive += 1
                elif matching_rows[0]['answer'] is True:
                    true_positive += 1
                elif matching_rows[0]['answer'] is False:
                    false_positive += 1

    query_num = len(checked_articles)
    no_answer_query = blind_negative + false_negative
    answer_query = query_num - no_answer_query
    print(f"For {query_num} queries total, retriever answered {answer_query} and did not answer {no_answer_query}.")
    print(f"For no_answer_query, false_negative: {false_negative}, blind_negative: {blind_negative}")
    print(f"For answered query, true_positive: {true_positive}, false_positive: {false_positive}, blind_positive:{blind_positive}")

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative + blind_negative) if (true_positive + false_negative + blind_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"recall at {top_k}: {recall:.4f}")
    print(f"precision at {top_k}: {precision:.4f}")
    print(f"f1 at {top_k}: {f1:.4f}")
    return {"result_path": result_path, "recall": recall, "precision": precision, "f1": f1, "missed_pairs": missed_pairs}


# ---------------------------------------------------------------------------
# Benchmark Table Generation (nDCG@k, Recall@k, F1@k with Highlighting)
# ---------------------------------------------------------------------------
def highlight_table(table, header):
    # table: List[List[str or float]]
    # header: List[str]
    num_cols = len(header)
    col_values = [[] for _ in range(num_cols)]
    for row in table:
        for i, val in enumerate(row):
            try:
                col_values[i].append(float(val))
            except:
                col_values[i].append(None)
    # Find best and second-best index per column (index 0 is model name)
    best_idx = [None] * num_cols
    second_idx = [None] * num_cols
    for j in range(1, num_cols):
        vals = [(i, v) for i, v in enumerate(col_values[j]) if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
        if len(vals_sorted) > 0:
            best_idx[j] = vals_sorted[0][0]
        if len(vals_sorted) > 1:
            second_idx[j] = vals_sorted[1][0]
    # Apply style
    styled_table = []
    for i, row in enumerate(table):
        styled_row = []
        for j, val in enumerate(row):
            sval = str(val)
            if j == 0:
                styled_row.append(sval)
            elif best_idx[j] == i:
                styled_row.append(f"{BLUE_BOLD}{sval}{RESET}")
            elif second_idx[j] == i:
                styled_row.append(f"{UNDERLINE}{sval}{RESET}")
            else:
                styled_row.append(sval)
        styled_table.append(styled_row)
    return styled_table


def evaluate_benchmark_single(result_path, article_network):
    from src.methods.ReX.hybrid_result import build_true_dict, build_true_dict_by_test, compute_ndcg, classify_type

    top_k = [5, 10, 50]
    top_k0 = [50]

    true_dicts_with_known_conflicts = build_true_dict()
    true_dicts = build_true_dict_by_test()

    processed_query = set()
    results = []
    with open(result_path, 'r', encoding='utf-8') as result_file:
        for line in result_file:
            row = json.loads(line.strip())
            results.append(row)
            processed_query.add(row['article_to_check'])

    all_positives = 0
    for q in processed_query:
        all_positives += len(true_dicts.get(q, []))

    table_metrics = {k: {"nDCG": 0.0, "Recall": 0.0, "Retrieval F1": 0.0} for k in top_k}

    for k0 in top_k0:
        results_as_binary_markings = []
        all_instances = 0

        true_articles_count = 0
        for r in tqdm(results, desc=f"Evaluating {os.path.basename(result_path)}"):
            query = r["article_to_check"]
            instance = {"article_to_check": query, "articles_as_binary": [], "articles_as_types": []}

            true_articles = set(true_dicts.get(query, []))
            true_articles_with_known_conflicts = set(true_dicts_with_known_conflicts.get(query, []))

            true_articles_count += len(true_articles)
            for a in r["articles"]:
                if a in true_articles_with_known_conflicts and a not in true_articles:
                    continue

                instance["articles_as_binary"].append(1 if a in true_articles else 0)
                instance["articles_as_types"].append(
                    classify_type(query, a, article_network) if a in true_articles else "NONE"
                )
            all_instances += len(instance["articles_as_binary"])
            results_as_binary_markings.append(instance)

        for k in top_k:
            total_true_positives = 0
            total_retrieved = 0
            recall = 0
            recall_cap = 0
            total_ndcg = 0.0

            for r in results_as_binary_markings:
                binary_list = r["articles_as_binary"]
                topk_list = binary_list[:k]
                total_true_positives += sum(topk_list)
                total_retrieved += len(topk_list)
                denom = len(true_dicts.get(r['article_to_check'], []))
                if denom > 0:
                    recall += sum(topk_list) / denom
                    recall_cap += sum(topk_list) / min(k, denom)
                    ideal_binary = [1 for _ in range(denom)] + [0 for _ in range(max(0, k - denom))]
                else:
                    ideal_binary = [0] * k
                total_ndcg += compute_ndcg(binary_list, ideal_binary, k)

            n_samples = len(results_as_binary_markings) if len(results_as_binary_markings) > 0 else 1
            recall = recall / n_samples * 100
            recall_cap = recall_cap / n_samples * 100
            precision = (total_true_positives / total_retrieved * 100) if total_retrieved > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ndcg = total_ndcg / n_samples * 100

            table_metrics[k]["nDCG"] = ndcg
            table_metrics[k]["Recall"] = recall
            table_metrics[k]["Retrieval F1"] = f1

    model_name = os.path.basename(result_path).split(".")[0]
    row = [model_name]
    for metric in ["nDCG", "Recall", "Retrieval F1"]:
        for k in [5, 10, 50]:
            row.append(f"{table_metrics[k][metric]:.2f}")
    return row


def run_benchmark_table(input_dir="./outputs/retrieval_results/article_key/"):
    from src.methods.LawGNN.article_network.article_network import ArticleNetwork
    from tabulate import tabulate

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Benchmark input directory not found: {input_dir}")

    article_network = ArticleNetwork()
    paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")
    ]
    paths.sort()
    if not paths:
        print(f"[WARN] No .jsonl files found in {input_dir}")
        return

    K = [5, 10, 50]
    header = ["Model"] + [f"nDCG@{k}" for k in K] + [f"Recall@{k}" for k in K] + [f"F1@{k}" for k in K]
    table = []
    for p in paths:
        try:
            print(f"Processing: {p}")
            row = evaluate_benchmark_single(p, article_network)
            table.append(row)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue

    print("-" * (len(header) * 15))
    styled_table = highlight_table(table, header)
    print(tabulate(styled_table, headers=header, tablefmt="github"))
    print("-" * (len(header) * 15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval results against ground truth or run benchmark table generation")
    parser.add_argument("--mode", type=str, choices=["eval", "benchmark"], default="eval", help="Mode: 'eval' for precision/recall/F1 evaluation of results, 'benchmark' for multi-model nDCG table generation")
    parser.add_argument("--benchmark", action="store_true", help="Shortcut for --mode benchmark")
    parser.add_argument("--benchmark_dir", type=str, default="./outputs/retrieval_results/article_key/", help="Directory containing result .jsonl files for benchmark table mode")

    # Eval arguments
    parser.add_argument("--result_path", type=str, default="./outputs/retrieval_results", help="Path to retrieval result jsonl file or directory (e.g. ./outputs/my_experiment/baseline_gat_baseline_laws10000.jsonl or ./outputs/retrieval_results)")
    parser.add_argument("--ground_truth_path", type=str, default="./data/datasets/LACD-biclassification/train-test-divide/test.jsonl", help="Path to ground truth test.jsonl")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for recall/precision (default 5)")
    parser.add_argument("--biencoder_top_k", type=int, default=10, help="Biencoder top-K used during retrieval (for logging only)")
    parser.add_argument("--output_missed", type=str, default=None, help="Optional path to save missed_pairs.jsonl")
    parser.add_argument("--metrics_output", type=str, default=None, help="Optional path to save metrics JSON (e.g. ./outputs/grex-10k-gat/metrics.json); if not provided, metrics are only printed")
    args = parser.parse_args()

    if args.benchmark or args.mode == "benchmark":
        run_benchmark_table(args.benchmark_dir)
    else:
        rows = load_ground_truth(args.ground_truth_path)
        print(f"[INFO] loaded {len(rows)} ground truth rows from {args.ground_truth_path}")

        # resolve result files
        result_files = []
        if os.path.isdir(args.result_path):
            result_files = glob.glob(os.path.join(args.result_path, "*.jsonl"))
            if not result_files:
                raise FileNotFoundError(f"No jsonl files found in directory {args.result_path}")
        elif os.path.isfile(args.result_path):
            result_files = [args.result_path]
        else:
            # try glob pattern
            result_files = glob.glob(args.result_path)
            if not result_files:
                raise FileNotFoundError(f"Result path not found: {args.result_path}")

        all_missed = []
        all_metrics = []
        for rf in sorted(result_files):
            res = evaluate_single(rf, rows, top_k=args.top_k, biencoder_top_k=args.biencoder_top_k)
            all_missed.extend(res["missed_pairs"])
            all_metrics.append({k: v for k, v in res.items() if k != "missed_pairs"})

        if args.output_missed and all_missed:
            with open(args.output_missed, 'w', encoding='utf-8') as outfile:
                for pair in all_missed:
                    json.dump(pair, outfile, ensure_ascii=False)
                    outfile.write('\n')
            print(f"[INFO] saved {len(all_missed)} missed pairs to {args.output_missed}")

        if args.metrics_output:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)) or ".", exist_ok=True)
            with open(args.metrics_output, 'w', encoding='utf-8') as f:
                json.dump({"top_k": args.top_k, "ground_truth": args.ground_truth_path, "results": all_metrics, "args": vars(args)}, f, ensure_ascii=False, indent=2)
            print(f"[METRICS] saved to {args.metrics_output}")
