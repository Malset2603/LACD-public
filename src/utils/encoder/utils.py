import polars as pl
import numpy as np
from transformers.trainer_callback import TrainerCallback
import torch
from sklearn.metrics import roc_auc_score


MAX_TOKEN_LENGTH = 4096

import torch

class ThreeLayerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1=512, hidden_size_2=256):
        super(ThreeLayerClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size_1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_1, hidden_size_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_2, 1)
        )

    def forward(self, x):
        return self.model(x)

class TensorBoardCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)

def data_augmentation(df, data_aug = "augmentation"):
    augmented_df = df.clone()

    # swap article1 <-> article2
    swapped_df = df.select(
        [pl.col("article2").alias("article1"), pl.col("article1").alias("article2")]
        + [pl.col(c) for c in df.columns if c not in ("article1", "article2")]
    )
    augmented_df = pl.concat([augmented_df, swapped_df], how="vertical")

    # groupby article1: join article2 with '\n', any answer
    # Use Python loop for string join to avoid Polars str.join quirks on small data
    grouped = {}
    for row in augmented_df.iter_rows(named=True):
        k = row["article1"]
        if k not in grouped:
            grouped[k] = {"article2_list": [], "answer": False}
        grouped[k]["article2_list"].append(row["article2"])
        grouped[k]["answer"] = grouped[k]["answer"] or bool(row["answer"])
    merged_rows = [
        {"article1": k, "article2": "\n".join(v["article2_list"]), "answer": v["answer"]}
        for k, v in grouped.items()
    ]
    # Preserve other columns if present (e.g., case_idx) — fill with defaults
    extra_cols = [c for c in df.columns if c not in ("article1", "article2", "answer")]
    for r in merged_rows:
        for c in extra_cols:
            r[c] = None
    merged_df = pl.DataFrame(merged_rows) if merged_rows else pl.DataFrame(schema={c: df.schema[c] for c in df.columns})
    # Ensure column order matches augmented_df
    if merged_df.height > 0:
        # Reorder to match df columns
        merged_df = merged_df.select(df.columns)
        augmented_df = pl.concat([augmented_df, merged_df], how="vertical")

    # Use original df for sampling true rows
    true_rows = df.filter(pl.col("answer") == True)
    # Collect all article texts for random sampling
    all_texts = df["article1"].to_list() + df["article2"].to_list()
    additional_rows = []

    for row in true_rows.iter_rows(named=True):
        random_value = np.random.choice(all_texts)
        new_row = row.copy()
        new_row["article2"] = "{}\n{}".format(row["article2"], random_value)
        additional_rows.append(new_row)

    for row in true_rows.iter_rows(named=True):
        random_value = np.random.choice(all_texts)
        new_row = row.copy()
        new_row["article1"] = "{}\n{}".format(row["article1"], random_value)
        additional_rows.append(new_row)

    if additional_rows:
        additional_df = pl.DataFrame(additional_rows)
        # Ensure schema matches
        additional_df = additional_df.select(df.columns) if set(additional_df.columns) == set(df.columns) else additional_df
        augmented_df = pl.concat([augmented_df, additional_df], how="vertical")

    return augmented_df

def balance_dataframe(df, column="answer"):
    # Sample to balance classes
    true_count = df.filter(pl.col(column) == True).height
    false_count = df.filter(pl.col(column) == False).height

    if true_count == false_count or true_count == 0 or false_count == 0:
        return df

    if true_count > false_count:
        # keep all false, sample true down to false_count
        false_df = df.filter(pl.col(column) == False)
        true_df = df.filter(pl.col(column) == True)
        # sample true
        true_df = true_df.sample(n=false_count, seed=42, shuffle=True)
        balanced_df = pl.concat([false_df, true_df], how="vertical")
    else:
        true_df = df.filter(pl.col(column) == True)
        false_df = df.filter(pl.col(column) == False)
        false_df = false_df.sample(n=true_count, seed=42, shuffle=True)
        balanced_df = pl.concat([true_df, false_df], how="vertical")

    # shuffle final
    return balanced_df.sample(fraction=1.0, shuffle=True, seed=42)

def accuracy_score(y_true, y_pred):
    # print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')
    return (y_true == y_pred).mean()

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    denom = true_positive + false_negative
    return true_positive / denom if denom > 0 else 0.0

def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    denom = true_positive + false_positive
    return true_positive / denom if denom > 0 else 0.0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denom = precision + recall
    return 2 * (precision * recall) / denom if denom > 0 else 0.0


def compute_metrics(p):
    from src.utils.encoder.utils import accuracy_score, f1_score, precision_score, recall_score
    # from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    pred_logits, labels = p

    # pred_logits가 (batch_size, 2)인 경우, 두 번째 확률 값만 사용
    pred_proba = torch.sigmoid(torch.tensor(pred_logits)).numpy()

    # 두 번째 값 (P(클래스 1))만 사용
    if pred_proba.shape[1] == 2:  # pred_proba가 (batch_size, 2)인 경우에만
        pred_proba = pred_proba[:, 1]  # (batch_size,)로 변환

    # Convert probabilities to class predictions
    pred = (pred_proba >= 0.5).astype(int).flatten()
    labels = labels.astype(int)

    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    
    # ROC AUC calculation using probabilities for the positive class
    # Handle single-class case (e.g., debug with 5 samples) where ROC AUC is undefined
    try:
        roc_auc = roc_auc_score(labels, pred_proba.flatten())
    except ValueError:
        roc_auc = 0.5
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }