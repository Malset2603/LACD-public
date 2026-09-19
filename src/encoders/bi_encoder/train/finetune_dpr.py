
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils.utils import seed_everything, SEED
from src.utils.encoder.biencoder_utils import DPRBiEncoderModel  # 새로 추가된 DPRBiEncoderModel
from src.utils.encoder.utils import MAX_TOKEN_LENGTH, TensorBoardCallback

from sklearn.metrics import accuracy_score, precision_recall_fscore_support



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LACD_DPR_PATH    = "/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-dpr/"
LACD_BI_PATH     = "/mnt/disk1/ugr/anseon2001/LACD/data/datasets/LACD-biclassification/train-test-divide-refine/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestBCDataset(Dataset):
    """article1, article2, answer 로드하는 이진분류용 Dataset"""
    def __init__(self, path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append({
                    "query":    obj["article1"],
                    "passage":  obj["article2"],
                    "label":    int(obj["answer"])  # 0 or 1
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        q = self.tokenizer(
            ex["query"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        p = self.tokenizer(
            ex["passage"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze batch dim
        return {
            "q_input_ids":       q["input_ids"].squeeze(0),
            "q_attention_mask":  q["attention_mask"].squeeze(0),
            "p_input_ids":       p["input_ids"].squeeze(0),
            "p_attention_mask":  p["attention_mask"].squeeze(0),
            "labels":            torch.tensor(ex["label"], dtype=torch.long)
        }

class DPRDataset(Dataset):
    """
    요구 JSONL 포맷 예시:
    {
      "query": "질의 텍스트",
      "positive": "정답 패시지 텍스트",
      "negatives": ["부정 1", "부정 2", ...]
    }
    """
    def __init__(self, path, tokenizer, max_length, num_negatives=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives

        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # neg 리스트가 짧으면 truncate or pad
                negs = obj.get("negatives", [])[:self.num_negatives]
                if len(negs) < self.num_negatives:
                    # 부족하면 그냥 복제
                    negs += [negs[0]] * (self.num_negatives - len(negs))
                self.examples.append({
                    "query": obj["query"],
                    "positive": obj["positive"],
                    "negatives": negs
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # 토크나이즈
        q = self.tokenizer(
            ex["query"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        pos = self.tokenizer(
            ex["positive"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg = [ self.tokenizer(
                    n,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
               ) for n in ex["negatives"] ]

        # 하나의 tensor 로 묶기: [1 + num_neg, seq_len]
        input_ids_ctx = torch.cat([pos["input_ids"]] + [n["input_ids"] for n in neg], dim=0)
        attn_mask_ctx = torch.cat([pos["attention_mask"]] + [n["attention_mask"] for n in neg], dim=0)

        return {
            "q_input_ids":    q["input_ids"].squeeze(0),
            "q_attention_mask": q["attention_mask"].squeeze(0),
            "p_input_ids":    input_ids_ctx,      # shape [num_ctx, seq]
            "p_attention_mask": attn_mask_ctx,    # shape [num_ctx, seq]
            "labels":         torch.tensor(0, dtype=torch.long)
            # positive 가 항상 index 0 에 위치 → CrossEntropyLoss 에서 정답 idx=0
        }

def data_collator(batch):
    """
    batch: list of dict
    → 각 필드 stacking
    """
    batch_q_ids   = torch.stack([b["q_input_ids"] for b in batch])
    batch_q_mask  = torch.stack([b["q_attention_mask"] for b in batch])

    # passages: [batch, num_ctx, seq_len]
    batch_p_ids   = torch.stack([b["p_input_ids"] for b in batch])
    batch_p_mask  = torch.stack([b["p_attention_mask"] for b in batch])

    labels = torch.stack([b["labels"] for b in batch])

    return {
        "q_input_ids":      batch_q_ids,
        "q_attention_mask": batch_q_mask,
        "p_input_ids":      batch_p_ids,
        "p_attention_mask": batch_p_mask,
        "labels":           labels,
    }

def bc_collator(batch):
    return {
        "q_input_ids":      torch.stack([b["q_input_ids"] for b in batch]),
        "q_attention_mask": torch.stack([b["q_attention_mask"] for b in batch]),
        "p_input_ids":      torch.stack([b["p_input_ids"] for b in batch]).unsqueeze(1),  # [B, 1, L]
        "p_attention_mask": torch.stack([b["p_attention_mask"] for b in batch]).unsqueeze(1),
        "labels":           torch.stack([b["labels"] for b in batch]),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   default="klue/roberta-base")
    parser.add_argument("--mode",            type=str,   default="train", help="train or test")
    parser.add_argument("--model_save_path", type=str,   default="./outputs/dpr")
    parser.add_argument("--tag",             type=str,   default="roberta-dpr-baseline")
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--num_negatives",   type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=SEED)
    args = parser.parse_args()

    seed_everything(args.seed)

    # tokenizer + max_length
    tokenizer = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    max_len = min(MAX_TOKEN_LENGTH, tokenizer.model_max_length)

    # 데이터셋 로드
    train_ds = DPRDataset(LACD_DPR_PATH+"train.jsonl", tokenizer, max_len, args.num_negatives)
    val_ds   = DPRDataset(LACD_DPR_PATH+"val.jsonl",   tokenizer, max_len, args.num_negatives)
    test_ds  = DPRDataset(LACD_DPR_PATH+"test.jsonl",  tokenizer, max_len, args.num_negatives)

    # 모델 초기화
    model = DPRBiEncoderModel(args.model).to(device)
    
    
    # Trainer 세팅
    training_args = TrainingArguments(
        output_dir      = os.path.join(args.model_save_path, args.tag),
        num_train_epochs= args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        evaluation_strategy = "epoch",
        save_strategy       = "epoch",
        logging_strategy    = "steps",
        logging_steps       = 50,
        load_best_model_at_end = True,
        metric_for_best_model   = "loss",
        greater_is_better       = False,
    )

    if args.mode == "train":
        writer = SummaryWriter(log_dir=os.path.join(args.model_save_path, args.tag, "tb"))
        trainer = Trainer(
            model            = model,
            args             = training_args,
            train_dataset    = train_ds,
            eval_dataset     = val_ds,
            data_collator    = data_collator,
            callbacks        = [TensorBoardCallback(writer)],
        )
        trainer.train()
        trainer.save_model(os.path.join(args.model_save_path, args.tag))
        tokenizer.save_pretrained(os.path.join(args.model_save_path, args.tag))
        writer.close()

        # 최종 test 평가
        metrics = trainer.evaluate(test_ds)
        print("Test results:", metrics)

        # 모델 파라미터와 토크나이저 저장
        torch.save(model, "./data/models/dpr/model.pth")
        tokenizer.save_pretrained("./data/models/dpr")

    else:  # ====== test 모드: biclassification ======
        test_path = os.path.join(LACD_BI_PATH, "test.jsonl")
        test_ds   = TestBCDataset(test_path, tokenizer, max_len)

        test_loader = DataLoader(
            test_ds,
            batch_size = 1,
            shuffle=False,
            collate_fn=bc_collator,
        )

        model.eval()
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for batch in test_loader:
                # 1) labels 분리
                labels = batch.pop("labels").to(device)
                # 2) 나머지 텐서를 device로 이동
                batch = {k: v.to(device) for k, v in batch.items()}

                # 3) question embedding 계산
                #    batch["q_input_ids"]: [B, L], q_attention_mask: [B, L]
                q_emb = model.get_question_embedding(
                    input_ids=batch["q_input_ids"],
                    attention_mask=batch["q_attention_mask"]
                )  # → [B, H]

                # 4) passage embedding 계산
                #    batch["p_input_ids"]: [B, 1, L]  (biclassification에서는 num_ctx=1)
                #    그래서 우선 차원 축소
                p_ids  = batch["p_input_ids"].squeeze(1)       # [B, L]
                p_mask = batch["p_attention_mask"].squeeze(1)  # [B, L]
                p_emb = model.get_passage_embedding(
                    input_ids=p_ids,
                    attention_mask=p_mask
                )  # → [B, H]

                # 5) dot‐product score 계산
                #    scores: [B] (num_ctx=1 이므로)
                scores = torch.einsum('bh,bh->b', q_emb, p_emb)

                # 6) 확률로 변환
                #    single‐logit → sigmoid
                probs = torch.sigmoid(scores)

                all_labels.extend(labels.cpu().tolist())
                all_probs .extend(probs.cpu().tolist())

        # 7) threshold=0.5 이진 예측 및 평가
        y_pred = [1 if p >= 0.5 else 0 for p in all_probs]

        acc       = accuracy_score(all_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, y_pred, average="binary"
        )

        print(f"Test on biclassification set ({len(all_labels)} samples):")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1       : {f1:.4f}")

