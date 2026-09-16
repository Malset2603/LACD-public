import argparse
import pandas as pd
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


from FlagEmbedding import FlagReranker

# Import your helper function and rule generation function
from src.utils.utils import article_key_function

# For rule-based methods, we import generate_rule from your prompt module.
# (Make sure this module is in your PYTHONPATH)
from src.methods.rule_augmentation.prompt import generate_rule

def load_test_data(dataset_path: str) -> pd.DataFrame:
    """
    Load the test dataset from a JSONL file.
    Each line is expected to have keys: "query", "article", and "answer".
    """
    test_file = dataset_path + "test.jsonl"
    df = pd.read_json(test_file, lines=True)
    return df

def prepare_input_pair(row: pd.Series, method: str):
    """
    Given a row from the test DataFrame and a method, return the input pair(s)
    for the cross encoder.
    
    For methods other than "ruleaug-divide", returns a tuple: (premise, hypothesis).
    For "ruleaug-divide", returns a tuple of two pairs:
       ( (premise, baseline_hypothesis), (premise, rule_hypothesis) )
    where rule_hypothesis is constructed using generate_rule.
    """
    premise = row["query"]
    hypothesis = row["article"]
    
    if method == "without-article":
        # Use the article key as hypothesis.
        hypothesis = article_key_function(hypothesis)
    elif method == "article-only":
        # Use the article text as is.
        pass
    elif method == "rule-only":
        # Use only the rule generated from the article.
        hypothesis = "\nrule:\n" + generate_rule(None, None, hypothesis)
    elif method == "ruleaug":
        # Append the rule to the original article.
        hypothesis = hypothesis + "\nrule:\n" + generate_rule(None, None, hypothesis)
    elif method == "ruleaug-divide":
        # For ruleaug-divide, we generate two pairs:
        # One using the original article (baseline) and one using the rule-only version.
        baseline_pair = (premise, hypothesis)
        rule_pair = (premise, "\nrule:\n" + generate_rule(None, None, hypothesis))
        return baseline_pair, rule_pair

    return (premise, hypothesis)

def evaluate(model: FlagReranker, test_df: pd.DataFrame, method: str, threshold: float = 0.5) -> dict:
    """
    Evaluate the cross encoder model on the test data.
    
    For methods other than "ruleaug-divide", each row is converted into a single pair.
    For "ruleaug-divide", two pairs are created for each row; the final score is the average.
    
    The model returns a continuous score for each pair. Scores above the threshold are
    predicted as positive (1), otherwise negative (0). Metrics are computed based on these.
    """
    pairs = []
    ground_truth = []

    print("Preparing input pairs ...")
    # Iterate through test rows and prepare pairs according to the method.
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Ground truth label: assume "answer" is boolean.
        gt = 1 if row["answer"] else 0
        
        if method == "ruleaug-divide":
            baseline_pair, rule_pair = prepare_input_pair(row, method)
            # Append both pairs along with the same ground-truth.
            pairs.append((baseline_pair, rule_pair))
        else:
            pair = prepare_input_pair(row, method)
            pairs.append(pair)
        ground_truth.append(gt)
    
    preds = []
    scores_all = []
    print("Running predictions ...")
    # Process the pairs: if ruleaug-divide, compute average score over two pairs.
    if method == "ruleaug-divide":
        for baseline_pair, rule_pair in tqdm(pairs):
            # Predict returns a list; here we pass one pair at a time.
            score_baseline = model.compute_score([baseline_pair])[0]
            score_rule = model.compute_score([rule_pair])[0]
            avg_score = (score_baseline + score_rule) / 2
            scores_all.append(avg_score)
            preds.append(1 if avg_score > threshold else 0)
    else:
        # For other methods, simply predict on the single pair.
        pair_list = [pair for pair in pairs]  # list of (premise, hypothesis)
        scores_all = [float(score) for score in model.compute_score(pair_list)]
        preds = [1 if score > threshold else 0 for score in scores_all]
    
    # Convert ground truth to int list.
    true_labels = [int(gt) for gt in ground_truth]
    
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    roc_auc = roc_auc_score(true_labels, scores_all) if len(np.unique(true_labels)) > 1 else None
    
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Testing for Contradiction Detection with Rule Augmentation")
    parser.add_argument("--model", type=str, default='BAAI/bge-reranker-v2-m3',
                        help="Path or name of the pre-trained cross encoder model.")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "without-article", "article-only", "rule-only", "ruleaug", "ruleaug-divide"],
                        help="Method to use for constructing hypothesis.")
    parser.add_argument("--dataset_path", type=str,
                        default="/mnt/disk1/anseon2001/LACD/data/database/lbox_open_statute_classification/",
                        help="Path to the dataset directory containing test.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold to convert model outputs to binary predictions")
    args = parser.parse_args()
    
    # Load test data.
    test_df = load_test_data(args.dataset_path)
    print(f"Loaded {len(test_df)} test samples.")
    if "rule" in args.method:
        from src.methods.rule_augmentation.prompt import rule_cache_start, rule_cache_end
        rule_cache_start(cache_path="./data/database/generated_rule_cache/prompt-base-full-law/rule_cache.jsonl")
    # Initialize the cross encoder.
    # Automatic padding is handled internally.
    model = FlagReranker(args.model)
    
    # Evaluate the model.
    metrics = evaluate(model, test_df, method=args.method, threshold=args.threshold)
    
    print("Zero-shot evaluation metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC:   {metrics['roc_auc']:.1%}")
    else:
        print("ROC AUC:   N/A (only one class present)")