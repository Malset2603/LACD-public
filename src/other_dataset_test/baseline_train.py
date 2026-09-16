# contradiction detection model 을 훈련하기 위한 코드
# 여기에서는 Full-fine-tune 만을 다룬다.

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter # type: ignore
from transformers import DataCollatorWithPadding
from src.other_dataset_test.model import CrossEncoderModel, RuleCrossEncoderModel
from src.utils.encoder.utils import data_augmentation, balance_dataframe, MAX_TOKEN_LENGTH, compute_metrics, TensorBoardCallback, get_class_weights, CustomTrainer
from src.utils.utils import seed_everything, SEED

from src.other_dataset_test.dataset import NLIDataset



if __name__ == "__main__":

    DATASET_PATH = "/mnt/disk1/anseon2001/LACD/data/database/lbox_open_statute_classification/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base")
    parser.add_argument("--rule_model", type=str, default="monologg/kobigbird-bert-base")
    parser.add_argument("--mode", type=str, help="train, test, or inference", default="train")
    
    # False, trainCivil, trainCriminal
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--data_augmentation", type=str, help="None, augmentation, noA, noB, noswap, or nobalance", default="None")

    parser.add_argument("--case_multiplier", type = int, default=1)
    parser.add_argument("--epoch", type = int, default=1)

    parser.add_argument("--method", type=str, help="without-article, article-only, rule-only, ruleaug or ruleaug-divide", default="baseline")
    parser.add_argument("--seed",type=int,help="seed",default=42)


    args = parser.parse_args()

    seed_everything(args.seed)

    case_multiplier = args.case_multiplier
    model_name = args.model
    rule_model_name = args.rule_model
    tag = args.tag
    epoch_num = args.epoch
    
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.config.pad_token_id = model.config.eos_token_id

    model = CrossEncoderModel(model_name=model_name)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tokenizer = model.tokenizer
    model.encoder.resize_token_embeddings(len(tokenizer))

    # tokenizer.padding_side = "right"

    # 분할된 인덱스를 사용하여 train, test 데이터프레임 생성
    train_df = pd.read_json(DATASET_PATH+ 'train.jsonl', lines=True)
    test_df = pd.read_json(DATASET_PATH+'test.jsonl', lines=True)
    val_df = pd.read_json(DATASET_PATH+'val.jsonl', lines=True)


    # train_df에 'case_idx' 컬럼을 추가하고 모든 값을 0으로 설정

    original_train_df = train_df.copy()

            
    max_length = min([MAX_TOKEN_LENGTH, tokenizer.model_max_length])

    train_dataset = NLIDataset(train_df, tokenizer, max_length=max_length, method=args.method)
    test_dataset = NLIDataset(test_df, tokenizer, max_length=max_length, method=args.method)
    val_dataset = NLIDataset(val_df, tokenizer, max_length=max_length, method=args.method)

    print(f"Token limit for {args.model}: {tokenizer.model_max_length}")
    print(f"Max length {args.model}: {max_length}")
    train_dataset.print_label_counts()
    test_dataset.print_label_counts()
    val_dataset.print_label_counts()
    
    training_args = TrainingArguments(
        output_dir=f'./outputs/lbox/small-fine-tune/{tag}',  # output directory
        num_train_epochs=epoch_num,                          # total number of training epochs
        per_device_train_batch_size=8,               # batch size for training
        per_device_eval_batch_size=8,                # batch size for evaluation
        warmup_steps=500,                            # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/lbox/small-fine-tune/{tag}',  # directory for storing logs
        # logging_steps=10,
        eval_strategy="epoch",                       # evaluation strategy
        eval_steps=1,                               # evaluation interval
        save_strategy="epoch",                       # save strategy to match eval steps
        save_steps=1,                               # save interval matching eval steps
        save_total_limit=1,                          # only keep the best model
        load_best_model_at_end=True,                 # load the best model at the end
        metric_for_best_model="eval_loss",           # metric to use for model selection
        report_to="tensorboard"                      # report to TensorBoard
    )
    if args.mode == "train":
        print('train')

        writer = SummaryWriter()

        # 클래스 가중치 계산
        # 클래스 가중치 및 num_labels 계산
        class_weights = get_class_weights(train_df)
        model.weight = class_weights[1] # type: ignore

        # 기존 Trainer → CustomTrainer 로 변경
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(writer)],
            class_weights=class_weights,  # 가중치 전달
        )

        trainer.train()

        test_results = trainer.evaluate(eval_dataset=test_dataset)

         # 결과를 저장
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)

        # 결과를 출력
        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        # TensorBoard에 기록
        writer.add_scalar("Test/F1", test_f1)
        writer.add_scalar("Test/Accuracy", test_accuracy)
        writer.add_scalar("Test/ROC_AUC", test_roc_auc)

        if args.model_save_path == "None":
            path = f"./data/models/lbox/{tag}"
        else:
            path = args.model_save_path
        
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, path+"/model.pth")
        tokenizer.save_pretrained(path)
            
    elif args.mode == "test":
        print("evaluate")
        model = torch.load(f"./data/models/lbox/{tag}/model.pth")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            compute_metrics=compute_metrics
        )
        eval_result = trainer.evaluate()

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

    
    if "rule" in args.method:

        rule_cache_end()
