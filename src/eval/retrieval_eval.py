import json
import argparse
import os
import glob
from src.utils.utils import article_key_function
from tqdm import tqdm

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
                line=line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    # also include missed_pairs if exists (legacy)
    legacy_missed = "./data/datasets/LACD-retrieval/missed_pairs.jsonl"
    if os.path.exists(legacy_missed) and legacy_missed not in paths:
        with open(legacy_missed, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
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
            line=line.strip()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval results against ground truth")
    parser.add_argument("--result_path", type=str, default="./outputs/retrieval_results", help="Path to retrieval result jsonl file or directory (e.g. ./outputs/my_experiment/baseline_gat_baseline_laws10000.jsonl or ./outputs/retrieval_results)")
    parser.add_argument("--ground_truth_path", type=str, default="./data/datasets/LACD-biclassification/train-test-divide/test.jsonl", help="Path to ground truth test.jsonl")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for recall/precision (default 5)")
    parser.add_argument("--biencoder_top_k", type=int, default=10, help="Biencoder top-K used during retrieval (for logging only)")
    parser.add_argument("--output_missed", type=str, default=None, help="Optional path to save missed_pairs.jsonl")
    parser.add_argument("--metrics_output", type=str, default=None, help="Optional path to save metrics JSON (e.g. ./outputs/grex-10k-gat/metrics.json); if not provided, metrics are only printed")
    args = parser.parse_args()

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
