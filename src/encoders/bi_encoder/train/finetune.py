# Bi-encoder NLI model using cosine similarity, adjusted for LACD-bi with the specified compute_metrics function.

import pandas as pd
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import argparse
from torch.utils.tensorboard import SummaryWriter # type: ignore
from src.utils.utils import article_key_function, seed_everything, SEED, LACD_DATASET_PATH

from src.utils.encoder.biencoder_utils import (
    BiEncoderModel,
)

from src.utils.encoder.utils import MAX_TOKEN_LENGTH, TensorBoardCallback, compute_metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base")
    parser.add_argument("--mode", type=str, help="train, test, or inference", default="train")

    # False, trainCivil, trainCriminal
    # parser.add_argument("--split_by_law", type=str, default="False")
    parser.add_argument("--model_save_path",type=str,help="a path for saving model. do not use None as the name",default="None",)
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")

    parser.add_argument("--case_multiplier", type=int, default=2)
    parser.add_argument("--epoch", type = int, default=5)

    parser.add_argument("--method", type=str, help="baseline or case-augmentation or case-concat-augmentation or rule-augmentation", default="baseline")

    # biencoder method
    parser.add_argument("--biencoder_method",type=str,help="cosine or linear",default="cosine")
    parser.add_argument("--seed",type=int,help="seed",default=42)

    args = parser.parse_args()

    seed_everything(args.seed)


    case_multiplier = args.case_multiplier
    model_name = args.model
    tag = args.tag
    biencoder_method = args.biencoder_method
    epoch = args.epoch

    # Initialize the bi-encoder model and tokenizer
    model = BiEncoderModel(model_name, method=biencoder_method)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.encoder.config.pad_token_id = tokenizer.pad_token_id


    train_df = pd.read_json(LACD_DATASET_PATH+'train.jsonl', lines=True)
    test_df = pd.read_json(LACD_DATASET_PATH+'test.jsonl', lines=True)
    val_df = pd.read_json(LACD_DATASET_PATH+'val.jsonl', lines=True)


    # Add 'case_idx' column to DataFrames
    if "case" in args.method:
        train_df['case_idx'] = 0
        test_df['case_idx'] = 0
        val_df['case_idx'] = 0
    if "rule" in args.method:
        train_df['rule_idx'] = 0
        test_df['rule_idx'] = 0
        val_df['rule_idx'] = 0


    original_train_df = train_df.copy()
    if "case" in args.method and case_multiplier > 1:
        for i in range(1, case_multiplier):
            temp_df = original_train_df.copy()
            temp_df["case_idx"] = i
            train_df = pd.concat([train_df, temp_df], ignore_index=True)

    elif "rule" in args.method and case_multiplier > 1:
        # 1부터 case_multiplier-1까지 반복
        for i in range(1, case_multiplier):
            # train_df를 복제
            temp_df = original_train_df.copy()
            # 복제한 데이터프레임의 'case_idx' 값을 i로 설정
            temp_df['rule_idx'] = i
            # 복제한 데이터프레임을 원래 데이터프레임에 붙여넣기
            train_df = pd.concat([train_df, temp_df], ignore_index=True)

    # Modify the NLIDataset to work with your data format
    class NLIDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length, method):
            self.dataframe = dataframe.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.method = method
            # Adjust label mapping: True -> 1, False -> 0
            self.label_map = {True: 1, False: 0}

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            premise = self.dataframe.iloc[index]["article1"]
            hypothesis = self.dataframe.iloc[index]["article2"]
            label = self.dataframe.iloc[index]["answer"]
            label = 1 if label else 0  # True -> 1, False -> 0
            

            encoding_a = self.tokenizer.encode_plus(
                text=premise,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoding_b = self.tokenizer.encode_plus(
                text=hypothesis,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            item = {
                "input_ids_a": encoding_a["input_ids"].squeeze(),
                "attention_mask_a": encoding_a["attention_mask"].squeeze(),
                "input_ids_b": encoding_b["input_ids"].squeeze(),
                "attention_mask_b": encoding_b["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.float),  # Float tensor for BCEWithLogitsLoss
            }
            return item

        def print_label_counts(self):
            label_counts = self.dataframe["answer"].value_counts().to_dict()
            print(f"Label distribution: {label_counts}")

    max_length = min([MAX_TOKEN_LENGTH, tokenizer.model_max_length])

    
    train_dataset = NLIDataset(
        train_df, tokenizer, max_length=max_length, method=args.method
    )
    test_dataset = NLIDataset(
        test_df, tokenizer, max_length=max_length, method=args.method
    )
    val_dataset = NLIDataset(
        val_df, tokenizer, max_length=max_length, method=args.method
    )

    print(f"Token limit for {args.model}: {tokenizer.model_max_length}")
    # train_dataset.print_label_counts()
    # test_dataset.print_label_counts()
    # val_dataset.print_label_counts()

    training_args = TrainingArguments(
        output_dir=f"./outputs/LACD-bi/small-fine-tune/{tag}",  # 모델 출력 경로
        num_train_epochs=epoch,  # 전체 epoch 수
        per_device_train_batch_size=4,  # 훈련 배치 크기
        per_device_eval_batch_size=4,  # 평가 배치 크기
        warmup_steps=500,  # Warmup 단계
        weight_decay=0,  # Weight decay
        logging_dir=f"./outputs/LACD-bi/small-fine-tune/{tag}",  # 로깅 경로
        logging_steps=10,  # 로깅 간격
        evaluation_strategy="epoch",  # 매 epoch마다 평가
        save_strategy="epoch",  # 매 epoch마다 저장
        save_total_limit=1,  # 저장 모델 수 제한
        load_best_model_at_end=True,  # 최적 모델 로드
        metric_for_best_model="eval_roc_auc",  # roc_auc를 최적화 지표로 사용
        greater_is_better=True,  # 지표가 클수록 좋은 모델
        report_to="tensorboard",  # TensorBoard로 로깅
    )

    

    if args.mode == "train":
        print("train")

        writer = SummaryWriter()

        # Adjust the data collator to handle outputs for classification
        def data_collator(features):
            batch = {
                "input_ids_a": torch.stack([f["input_ids_a"] for f in features]),
                "attention_mask_a": torch.stack([f["attention_mask_a"] for f in features]),
                "input_ids_b": torch.stack([f["input_ids_b"] for f in features]),
                "attention_mask_b": torch.stack([f["attention_mask_b"] for f in features]),
                "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float),
            }
            return batch

        trainer = Trainer(
            model=model,  # the bi-encoder model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(writer)],
        )

        trainer.train()

        test_results = trainer.evaluate(eval_dataset=test_dataset)

        # 결과를 저장
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)

        # 결과를 출력
        print(f"Test F1 Score: {test_f1}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test ROC AUC: {test_roc_auc}")

        # TensorBoard에 기록
        writer.add_scalar("Test/F1", test_f1)
        writer.add_scalar("Test/Accuracy", test_accuracy)
        writer.add_scalar("Test/ROC_AUC", test_roc_auc)

        if args.model_save_path == "None":
            path = f"./data/models/LACD-bi/{tag}"
        else:
            path = args.model_save_path

        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(model, path+"/model.pth")
        tokenizer.save_pretrained(path)

        writer.close()

    elif args.mode == "test":
        print("evaluate")
        model = torch.load(f"./data/models/LACD-bi/{tag}/model.pth")
        def data_collator(features):
            batch = {
                "input_ids_a": torch.stack([f["input_ids_a"] for f in features]),
                "attention_mask_a": torch.stack([f["attention_mask_a"] for f in features]),
                "input_ids_b": torch.stack([f["input_ids_b"] for f in features]),
                "attention_mask_b": torch.stack([f["attention_mask_b"] for f in features]),
                "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float),
            }
            return batch

        trainer = Trainer(
            model=model,  # the bi-encoder model
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        eval_result = trainer.evaluate()
        # print(f"Evaluation results: {eval_result}")

        # Print test results
        test_f1 = eval_result.get("eval_f1", 0)
        test_accuracy = eval_result.get("eval_accuracy", 0)
        test_roc_auc = eval_result.get("eval_roc_auc", 0)
        test_precision = eval_result.get("eval_recall", 0)
        test_recall = eval_result.get("eval_precision", 0)

        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test precision Score: {test_precision:.1%}")
        print(f"Test recall Score: {test_recall:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

    if args.method == "case-augmentation" or args.method == "case-concat-augmentation":
        case_cache_end()
    elif args.method == "rule-augmentation":
        rule_cache_end()
