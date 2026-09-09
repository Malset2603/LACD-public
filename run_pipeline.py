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
    result = subprocess.run(cmd)
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
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base", help="Bi/cross encoder model")
    parser.add_argument("--gnn_method", type=str, default="gat", choices=["gcn", "graphsage", "gat", "vanilla", "gathybrid", "graphsagehybrid"], help="GNN method")
    parser.add_argument("--epoch", type=int, default=3, help="Epochs for bi and cross training (cross uses this value, default 3 as in paper)")
    parser.add_argument("--bi_epoch", type=int, default=None, help="Epochs for bi-encoder only (default: --epoch)")
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length (default 4096 full, use 512 for mini VRAM)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K for retrieval and eval")
    # Phase 1a: bi-encoder loss selection (default keeps original GReX BCE)
    parser.add_argument("--biencoder_loss", type=str, default="bce", choices=["bce", "infonce"],
                        help="Loss for bi-encoder: 'bce' (original GReX, default) or 'infonce' (contrastive with temperature)")
    parser.add_argument("--infonce_tau", type=float, default=0.05,
                        help="Temperature for InfoNCE (only when --biencoder_loss=infonce, recommended 0.05)")
    parser.add_argument("--steps", type=str, default="all", help="Steps to run: 'all' or comma list like '1,2,3,4,5'")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs (default: overwrite)")

    args = parser.parse_args()
    steps = parse_steps(args.steps)

    # Derive consistent names
    bi_tag = args.bi_tag or f"kbb-{args.exp}"
    cross_tag = args.cross_tag or f"{bi_tag}-gat"
    chroma_name = args.chroma_name or bi_tag
    output_dir = args.output_dir or f"./outputs/{args.exp}"
    bi_epoch = args.bi_epoch if args.bi_epoch is not None else args.epoch

    # Metric/output paths (auto-created, overwritten if exists)
    metrics_bi = os.path.join(output_dir, "metrics-bi.json")
    metrics_cross = os.path.join(output_dir, "metrics-cross.json")
    metrics_retrieval = os.path.join(output_dir, "metrics-retrieval.json")
    summary_path = os.path.join(output_dir, "summary.json")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] exp={args.exp} bi_tag={bi_tag} cross_tag={cross_tag} chroma={chroma_name} output_dir={output_dir} steps={sorted(steps)}")

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
        ] + subset_args()
        if args.fp16:
            cmd.append("--fp16")
        run_cmd(cmd, dry_run=args.dry_run)

    # Step 2: Build Chroma DB
    if 2 in steps:
        cmd = [
            sys.executable, "./src/main.py",
            "--biencoder_model_path", f"./data/models/LACD-bi/{bi_tag}",
            "--chroma_db_name", chroma_name,
            "--biencoder_method", "baseline",
            "--retrieval_method", "bi-only",
        ] + subset_args()
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
            "--max_length", str(args.max_length),
            "--metrics_output", metrics_cross,
        ] + subset_args()
        if args.fp16:
            cmd.append("--fp16")
        run_cmd(cmd, dry_run=args.dry_run)

    # Step 4: Benchmark hybrid
    if 4 in steps:
        cmd = [
            sys.executable, "./src/main.py",
            "--crossencoder_model_path", f"./data/models/LACD-cross/gnns/{cross_tag}",
            "--biencoder_model_path", f"./data/models/LACD-bi/{bi_tag}",
            "--chroma_db_name", chroma_name,
            "--retrieval_method", "hybrid",
            "--crossencoder_index_method", args.gnn_method,
            "--crossencoder_method", "baseline",
            "--biencoder_method", "baseline",
            "--mode", "test-benchmark",
            "--output_dir", output_dir,
            "--crossencoder_top_k", str(args.top_k),
            "--biencoder_top_k", str(args.top_k),
        ] + subset_args()
        run_cmd(cmd, dry_run=args.dry_run)

    # Step 5: Eval
    if 5 in steps:
        cmd = [
            sys.executable, "src/eval/retrieval_eval.py",
            "--result_path", output_dir,
            "--top_k", str(args.top_k),
            "--metrics_output", metrics_retrieval,
        ]
        run_cmd(cmd, dry_run=args.dry_run)

    # Aggregate summary.json (even if some steps were skipped, collect what exists)
    if not args.dry_run:
        summary = {
            "exp": args.exp,
            "bi_tag": bi_tag,
            "cross_tag": cross_tag,
            "chroma_name": chroma_name,
            "output_dir": output_dir,
            "args": vars(args),
            "metrics": {},
        }
        for key, path in [("bi", metrics_bi), ("cross", metrics_cross), ("retrieval", metrics_retrieval)]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        summary["metrics"][key] = json.load(f)
                except Exception as e:
                    summary["metrics"][key] = {"error": str(e), "path": path}
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[SUMMARY] aggregated metrics -> {summary_path}")
        # Also print summary to stdout
        print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))

    print("\n[DONE] pipeline finished.")


if __name__ == "__main__":
    main()
