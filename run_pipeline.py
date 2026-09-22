#!/usr/bin/env python3
"""
GReX pipeline orchestrator — single entry point for the 5-step experiment.

Steps:
  1. Bi-encoder training
  2. Build Chroma DB
  3. Cross-encoder + GNN training
  4. Benchmark hybrid retrieval
  5. Evaluate retrieval metrics

All steps share consistent --subset_laws / --subset_ratio / tags to avoid
mismatched graphs. Outputs (including optional metrics JSON) are collected
under a single --output_dir and aggregated into summary.json.

Examples:
  # Mini (10k laws, fast)
  python run_pipeline.py --exp grex-10k --subset_laws 10000 --epoch 2 --max_length 512 --fp16

  # Full data
  python run_pipeline.py --exp grex-full --epoch 3

  # Only rerun retrieval + eval (e.g., after fixing thresholds)
  python run_pipeline.py --exp grex-10k --subset_laws 10000 --steps 4,5

  # Custom tags / output dir
  python run_pipeline.py --exp grex-10k --bi_tag kbb-grex-10k --cross_tag kbb-grex-gat-10k --output_dir ./outputs/grex-10k-gat

  # Multi-seed (e.g. paper mean+-std over 3 runs)
  python run_pipeline.py --exp grex-paper --epoch 10 --seed 0,1,2 --biencoder_top_k 100
  # Single seed stays exactly as before; multi-seed runs each seed with
  # suffixed tags (kbb-{exp}-s{seed}) and subdirs (./outputs/{exp}/seed{seed}),
  # then aggregates per-seed metrics plus mean/std into summary.json.
  # Chroma optimization: with --bi_epoch 0 (frozen bi-encoder, paper repro),
  # embeddings are identical across seeds so Step 2 builds the shared Chroma
  # DB once (no -s{seed} suffix) and later seeds reuse it; with --bi_epoch > 0
  # each seed gets its own Chroma DB.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, dry_run=False):
    print(f"\n[RUN] {' '.join(cmd)}")
    if dry_run:
        return 0
    env = os.environ.copy()
    # Ensure project root is on PYTHONPATH so `import src` works
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] command failed with exit code {result.returncode}: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def parse_steps(steps_str):
    if steps_str.strip().lower() == "all":
        return {1, 2, 3, 4, 5}
    try:
        return {int(s.strip()) for s in steps_str.split(",") if s.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError("--steps must be 'all' or comma-separated numbers like '1,2,3'")


def parse_seeds(seeds_str):
    """Parse --seed '42' or '0,1,2' -> [42] or [0, 1, 2] (deduped, order kept)."""
    try:
        seeds = [int(s.strip()) for s in str(seeds_str).split(",") if s.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError("--seed must be a single int (e.g. 42) or comma-separated ints (e.g. 0,1,2)")
    if not seeds:
        raise argparse.ArgumentTypeError("--seed must contain at least one int")
    seen = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _mean_std(vals):
    import statistics
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _agg_dicts(dicts):
    """Average a list of same-structured dicts -> (mean_dict, std_dict).

    Numeric leaves are averaged (sample stdev, 0.0 for n=1); lists are
    averaged element-wise; non-numeric leaves take the first value.
    """
    keys = set().union(*[d.keys() for d in dicts])
    mean_d, std_d = {}, {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        if len(vals) != len(dicts):
            mean_d[k] = vals[0] if vals else None
            std_d[k] = vals[0] if vals else None
        elif all(isinstance(v, bool) for v in vals):
            mean_d[k] = vals[0]
            std_d[k] = vals[0]
        elif all(isinstance(v, (int, float)) for v in vals):
            m, s = _mean_std([float(v) for v in vals])
            mean_d[k] = m
            std_d[k] = s
        elif all(isinstance(v, dict) for v in vals):
            m, s = _agg_dicts(vals)
            mean_d[k] = m
            std_d[k] = s
        elif all(isinstance(v, list) for v in vals) and all(len(v) == len(vals[0]) for v in vals):
            mean_l, std_l = [], []
            for i in range(len(vals[0])):
                elems = [v[i] for v in vals]
                if all(isinstance(e, bool) for e in elems):
                    mean_l.append(elems[0])
                    std_l.append(elems[0])
                elif all(isinstance(e, (int, float)) for e in elems):
                    m, s = _mean_std([float(e) for e in elems])
                    mean_l.append(m)
                    std_l.append(s)
                elif all(isinstance(e, dict) for e in elems):
                    m, s = _agg_dicts(elems)
                    mean_l.append(m)
                    std_l.append(s)
                else:
                    mean_l.append(elems[0])
                    std_l.append(elems[0])
            mean_d[k] = mean_l
            std_d[k] = std_l
        else:
            mean_d[k] = vals[0]
            std_d[k] = vals[0]
    return mean_d, std_d


def load_metrics_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="GReX pipeline orchestrator (mini and full)")
    parser.add_argument("--exp", type=str, required=True, help="Experiment base name, e.g. grex-10k or grex-full (used to derive tags and output dir)")
    parser.add_argument("--bi_tag", type=str, default=None, help="Bi-encoder tag (default: kbb-{exp})")
    parser.add_argument("--cross_tag", type=str, default=None, help="Cross-encoder tag (default: {bi_tag}-gat)")
    parser.add_argument("--chroma_name", type=str, default=None, help="Chroma DB name (default: {bi_tag})")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for retrieval results and metrics (default: ./outputs/{exp})")

    # shared subset / training args (optional, no default -> full data if omitted)
    parser.add_argument("--subset_laws", type=int, default=None, help="Subset laws for ArticleNetwork (e.g. 10000); omit for full 79k")
    parser.add_argument("--subset_ratio", type=float, default=None, help="Subset ratio for train/val/test stratified sampling (e.g. 0.15)")
    parser.add_argument("--subset_seed", type=int, default=42, help="Seed for subset sampling")
    parser.add_argument("--seed", type=parse_seeds, default=[42], help="Seed(s) for model training in Steps 1 & 3 (single 42 or comma list 0,1,2; multi-seed runs each seed and aggregates mean/std)")
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base", help="Bi/cross encoder model")
    parser.add_argument("--gnn_method", type=str, default="gat", choices=["gcn", "graphsage", "gat", "vanilla", "gathybrid", "graphsagehybrid"], help="GNN method")
    parser.add_argument("--epoch", type=int, default=3, help="Epochs for bi and cross training (cross uses this value, the default value is 3)")
    parser.add_argument("--bi_epoch", type=int, default=None, help="Epochs for bi-encoder only (default: --epoch)")
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length (default 4096 full, use 512 for mini VRAM)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training (default 4)")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for Chroma DB build and retrieval/eval (default: 32, decoupled from training --batch_size since inference needs no gradients)")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 (training Steps 1 & 3, plus retrieval encoding in Steps 2 & 4 via --fp16_eval)")
    parser.add_argument("--fp16_eval", action="store_true", help="Enable fp16 autocast for retrieval encoding in Steps 2 & 4 (CUDA only; verify recall parity first)")
    parser.add_argument("--force_rebuild", action="store_true", help="Wipe the Chroma collection and rebuild from scratch in Step 2 (use when content is suspect; partial DBs otherwise resume)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K cutoff for retrieval evaluation in Step 5")
    parser.add_argument("--eval_ks", type=str, default="5,10,50", help="Comma-separated cutoffs for nDCG/Recall/F1 in Step 5 (default: 5,10,50)")
    parser.add_argument("--biencoder_top_k", type=int, default=150, help="Top-K retrieval depth in Step 4 (paper Sec 4.2: 100 for GReX/ReX+Re2 -> ~150 after expand, 150 for Re2/Re2+LGNN; lower it, e.g. 5-10, for fast experiments)")
    # GReX / ReX expansion options (default: 'rex2' for full GReX method)
    parser.add_argument("--rex_method", type=str, default="rex2", choices=["rex2", "baseline", "rocchio"],
                        help="ReX expansion method: 'rex2' (GReX full method, default), 'baseline' (Re2+LGNN without expansion), or 'rocchio'")
    parser.add_argument("--rex_conflict", type=str, default="train", choices=["train", "train-generate"],
                        help="Conflict graph source for ReX expansion ('train' or 'train-generate')")
    parser.add_argument("--gnn_edge_way", type=str, default="both", choices=["both", "forward", "backward"],
                        help="Edge propagation direction in ArticleNetwork ('both', 'forward', 'backward')")
    # Phase 1a: bi-encoder loss selection (default keeps original GReX BCE)
    parser.add_argument("--biencoder_loss", type=str, default="bce", choices=["bce", "infonce"],
                        help="Loss for bi-encoder: 'bce' (original GReX, default) or 'infonce' (contrastive with temperature)")
    parser.add_argument("--infonce_tau", type=float, default=0.05,
                        help="Temperature for InfoNCE (only when --biencoder_loss=infonce, recommended 0.05)")
    parser.add_argument("--steps", type=str, default="all", help="Steps to run: 'all' or comma list like '1,2,3,4,5'")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()
    seeds = args.seed if isinstance(args.seed, list) else parse_seeds(args.seed)
    steps = parse_steps(args.steps)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else 32
    multi = len(seeds) > 1

    # Base names (without seed suffix)
    base_bi = args.bi_tag or f"kbb-{args.exp}"
    base_output = args.output_dir or f"./outputs/{args.exp}"
    bi_epoch = args.bi_epoch if args.bi_epoch is not None else args.epoch

    # Helper to build common subset args
    def subset_args():
        parts = []
        if args.subset_laws is not None:
            parts += ["--subset_laws", str(args.subset_laws)]
        if args.subset_ratio is not None:
            parts += ["--subset_ratio", str(args.subset_ratio)]
        if args.subset_seed != 42:
            parts += ["--subset_seed", str(args.subset_seed)]
        return parts

    def names_for(seed):
        """Per-seed tags/dirs. Single seed keeps legacy names exactly.

        Chroma is shared (no seed suffix) when bi_epoch==0 since the frozen
        bi-encoder is identical across seeds; per-seed otherwise.
        """
        if not multi:
            b = args.bi_tag or f"kbb-{args.exp}"
            c = args.cross_tag or f"{b}-gat"
            ch = args.chroma_name or b
            out = args.output_dir or f"./outputs/{args.exp}"
            return b, c, ch, out
        b = f"{base_bi}-s{seed}"
        c = f"{args.cross_tag}-s{seed}" if args.cross_tag else f"{b}-gat"
        if bi_epoch == 0:
            ch = args.chroma_name or base_bi
        else:
            ch = f"{(args.chroma_name or base_bi)}-s{seed}"
        out = os.path.join(base_output, f"seed{seed}")
        return b, c, ch, out

    def run_single(seed, bi_tag, cross_tag, chroma_name, output_dir, skip_step2=False):
        metrics_bi = os.path.join(output_dir, "metrics-bi.json")
        metrics_cross = os.path.join(output_dir, "metrics-cross.json")
        metrics_retrieval = os.path.join(output_dir, "metrics-retrieval.json")
        summary_path = os.path.join(output_dir, "summary.json")
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] exp={args.exp} seed={seed} bi_tag={bi_tag} cross_tag={cross_tag} chroma={chroma_name} rex_method={args.rex_method} batch_size={args.batch_size} eval_batch_size={eval_batch_size} output_dir={output_dir} steps={sorted(steps)}")

        # Step 1: Bi-encoder training
        if 1 in steps:
            cmd = [
                sys.executable, "./src/encoders/bi_encoder/train/finetune.py",
                "--model", args.model,
                "--mode", "train",
                "--tag", bi_tag,
                "--max_length", str(args.max_length),
                "--batch_size", str(args.batch_size),
                "--epoch", str(bi_epoch),
                "--metrics_output", metrics_bi,
                "--biencoder_loss", args.biencoder_loss,
                "--infonce_tau", str(args.infonce_tau),
                "--seed", str(seed),
            ] + subset_args()
            if args.fp16:
                cmd.append("--fp16")
            if args.gradient_checkpointing:
                cmd.append("--gradient_checkpointing")
            run_cmd(cmd, dry_run=args.dry_run)

        # Step 2: Build Chroma DB (skipped for seeds[1:] when shared: bi_epoch==0
        # yields identical frozen embeddings, so the first seed's build is reused)
        if 2 in steps and not skip_step2:
            cmd = [
                sys.executable, "./src/main.py",
                "--biencoder_model_path", f"./data/models/LACD-bi/{bi_tag}",
                "--chroma_db_name", chroma_name,
                "--retrieval_method", "retrieval",
                "--batch_size", str(eval_batch_size),
                "--max_length", str(args.max_length),
            ] + subset_args()
            if args.fp16 or args.fp16_eval:
                cmd.append("--fp16_eval")
            if args.force_rebuild:
                cmd.append("--force_rebuild")
            run_cmd(cmd, dry_run=args.dry_run)

        # Step 3: Cross-encoder + GNN
        if 3 in steps:
            cmd = [
                sys.executable, "./src/methods/LawGNN/train/crossencoder_finetune.py",
                "--tag", cross_tag,
                "--gnn_method", args.gnn_method,
                "--chroma_db_name", chroma_name,
                "--case_augmentation_method", "baseline",
                "--epoch", str(args.epoch),
                "--batch_size", str(args.batch_size),
                "--max_length", str(args.max_length),
                "--metrics_output", metrics_cross,
                "--seed", str(seed),
            ] + subset_args()
            if args.fp16:
                cmd.append("--fp16")
            if args.gradient_checkpointing:
                cmd.append("--gradient_checkpointing")
            run_cmd(cmd, dry_run=args.dry_run)

        # Step 4: Benchmark GReX retrieval (Rerank-then-Expand)
        if 4 in steps:
            cmd = [
                sys.executable, "./src/main.py",
                "--crossencoder_model_path", f"./data/models/LACD-cross/gnns/{cross_tag}",
                "--biencoder_model_path", f"./data/models/LACD-bi/{bi_tag}",
                "--chroma_db_name", chroma_name,
                "--retrieval_method", "re2",
                "--crossencoder_index_method", args.gnn_method,
                "--rex_method", args.rex_method,
                "--rex_conflict", args.rex_conflict,
                "--gnn_edge_way", args.gnn_edge_way,
                "--mode", "test-benchmark",
                "--output_dir", output_dir,
                "--batch_size", str(eval_batch_size),
                "--max_length", str(args.max_length),
                "--biencoder_top_k", str(args.biencoder_top_k),
            ] + subset_args()
            if args.fp16 or args.fp16_eval:
                cmd.append("--fp16_eval")
            # NOTE: never forward --force_rebuild here: Step 4 calls
            # binary_retriever per query, so a rebuild flag would wipe and
            # re-encode the entire DB on EVERY query. Rebuilds belong to Step 2.
            run_cmd(cmd, dry_run=args.dry_run)

        # Step 5: Eval
        if 5 in steps:
            cmd = [
                sys.executable, "src/eval/retrieval_eval.py",
                "--result_path", output_dir,
                "--top_k", str(args.top_k),
                "--top_ks", args.eval_ks,
                "--metrics_output", metrics_retrieval,
            ]
            run_cmd(cmd, dry_run=args.dry_run)

        # Per-seed summary.json (even if some steps were skipped, collect what exists)
        metrics = {}
        if not args.dry_run:
            for key, path in [("bi", metrics_bi), ("cross", metrics_cross), ("retrieval", metrics_retrieval)]:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            metrics[key] = json.load(f)
                    except Exception as e:
                        metrics[key] = {"error": str(e), "path": path}
            summary = {
                "exp": args.exp,
                "seed": seed,
                "seeds": seeds,
                "bi_tag": bi_tag,
                "cross_tag": cross_tag,
                "chroma_name": chroma_name,
                "output_dir": output_dir,
                "args": {**vars(args), "seed": seed, "seeds": seeds},
                "metrics": metrics,
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n[SUMMARY] seed {seed} aggregated metrics -> {summary_path}")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return metrics

    per_seed_metrics = {}
    per_seed_names = {}
    shared_chroma = multi and bi_epoch == 0
    if shared_chroma:
        print(f"[INFO] bi_epoch=0 with {len(seeds)} seeds: Chroma DB shared as '{names_for(seeds[0])[2]}', Step 2 runs once (seed {seeds[0]})")
    for i, seed in enumerate(seeds):
        bi_tag, cross_tag, chroma_name, output_dir = names_for(seed)
        per_seed_names[str(seed)] = {
            "bi_tag": bi_tag, "cross_tag": cross_tag,
            "chroma_name": chroma_name, "output_dir": output_dir,
        }
        skip_step2 = shared_chroma and i > 0
        if skip_step2:
            print(f"[INFO] seed {seed}: skipping Step 2 (reusing shared Chroma '{chroma_name}' from seed {seeds[0]})")
        per_seed_metrics[str(seed)] = run_single(seed, bi_tag, cross_tag, chroma_name, output_dir, skip_step2=skip_step2)

    # Aggregate summary.json across seeds (mean/std over numeric leaves)
    if not args.dry_run:
        if not multi:
            pass  # per-seed summary above is already the legacy summary.json
        else:
            os.makedirs(base_output, exist_ok=True)
            mean, std = {}, {}
            # bi / cross: average inner .metrics dicts
            for key in ("bi", "cross"):
                dicts = []
                for s in seeds:
                    m = per_seed_metrics.get(str(s), {}).get(key)
                    if isinstance(m, dict) and isinstance(m.get("metrics"), dict):
                        dicts.append(m["metrics"])
                if dicts:
                    m_mean, m_std = _agg_dicts(dicts)
                    mean[key], std[key] = m_mean, m_std
            # retrieval: average results[0] numeric fields (recall/precision/f1/multi_k)
            rdicts = []
            for s in seeds:
                m = per_seed_metrics.get(str(s), {}).get("retrieval")
                if isinstance(m, dict) and isinstance(m.get("results"), list) and m["results"]:
                    r0 = m["results"][0]
                    rdicts.append({k: v for k, v in r0.items() if k not in ("result_path", "missed_pairs")})
            if rdicts:
                r_mean, r_std = _agg_dicts(rdicts)
                mean["retrieval"], std["retrieval"] = r_mean, r_std
            summary = {
                "exp": args.exp,
                "seeds": seeds,
                "per_seed": {
                    str(s): {**per_seed_names[str(s)], "metrics": per_seed_metrics.get(str(s), {})}
                    for s in seeds
                },
                "mean": mean,
                "std": std,
                "output_dir": base_output,
                "args": {**vars(args), "seeds": seeds},
                "metrics_note": "mean/std use sample stdev (0.0 for n=1); retrieval averages results[0] numeric fields",
            }
            summary_path = os.path.join(base_output, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n[SUMMARY] multi-seed ({len(seeds)} seeds) aggregated metrics -> {summary_path}")
            print(json.dumps({"mean": mean, "std": std}, indent=2, ensure_ascii=False))

    print("\n[DONE] pipeline finished.")


if __name__ == "__main__":
    main()
