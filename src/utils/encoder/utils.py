import pandas as pd
import numpy as np
from transformers.trainer_callback import TrainerCallback
import torch
from sklearn.metrics import roc_auc_score


# MAX_TOKEN_LENGTH = 4096
MAX_TOKEN_LENGTH = 2048
# MAX_TOKEN_LENGTH = 1024

# MAX_TOKEN_LENGTH = 512

import torch
from torch.nn import BCEWithLogitsLoss


# 클래스별 answer 값의 분포 확인 및 가중치 계산
def get_class_weights(train_df):
    answer_counts = train_df["answer"].value_counts().sort_index().to_numpy()

    class_weights = 1.0 / answer_counts  # 반비례 가중치 계산
    class_weights = class_weights / class_weights.sum()  # 정규화
    print(class_weights)
    return torch.tensor(class_weights, dtype=torch.float).to("cuda")  # GPU 적용

# 가중치를 적용한 커스텀 Trainer
from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()  # BCEWithLogitsLoss는 float labels 필요
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze(-1)  # (batch_size, 1) → (batch_size,)

        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights[1])  # Binary classification에서 가중치 적용
        loss = loss_fct(logits, labels)  # BCEWithLogitsLoss 적용

        return (loss, outputs) if return_outputs else loss

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
    augmented_df = df.copy()

    # if data_aug != "noswap":

    swapped_df = df.copy()
    swapped_df[["article1", "article2"]] = swapped_df[["article2", "article1"]]
    augmented_df = pd.concat([augmented_df, swapped_df], ignore_index=True)

    merged_df = augmented_df.groupby("article1").agg({
        "article2": '\n'.join,
        "answer": 'any'
    }).reset_index()

    augmented_df = pd.concat([augmented_df, merged_df], ignore_index=True)

    true_rows = df[df["answer"] == True]
    additional_rows = []

    # if data_aug != "noB": #원래는 noA 였음.
    for idx, row in true_rows.iterrows():
        random_value = np.random.choice(df["article1"].tolist() + df["article2"].tolist())
        new_row = row.copy()
        new_row["article2"] = "{}\n{}".format(row["article2"], random_value)
        additional_rows.append(new_row)

    # if data_aug != "noA": #원래는 noB
    for idx, row in true_rows.iterrows():
        random_value = np.random.choice(df["article1"].tolist() + df["article2"].tolist())
        new_row = row.copy()
        new_row["article1"] = "{}\n{}".format(row["article1"], random_value)
        additional_rows.append(new_row)

    additional_df = pd.DataFrame(additional_rows)
    augmented_df = pd.concat([augmented_df, additional_df], ignore_index=True)

    return augmented_df

def balance_dataframe(df, column="answer"):
    true_count = df[df[column] == True].shape[0]
    false_count = df[df[column] == False].shape[0]

    if true_count > false_count:
        larger_group = True
        smaller_group = False
    else:
        larger_group = False
        smaller_group = True

    delete_count = abs(true_count - false_count)
    larger_group_indices = df[df[column] == larger_group].index
    drop_indices = np.random.choice(larger_group_indices, delete_count, replace=False)
    balanced_df = df.drop(drop_indices)

    return balanced_df

def accuracy_score(y_true, y_pred):
    # print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')
    return (y_true == y_pred).mean()

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


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
    roc_auc = roc_auc_score(labels, pred_proba.flatten())
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }