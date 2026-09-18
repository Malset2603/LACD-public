import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from transformers import DataCollatorWithPadding
from src.utils.encoder.crossencoder_utils import (
    CrossEncoderModel,
    NLIDataset,
)
from src.utils.encoder.utils import (
    MAX_TOKEN_LENGTH,
    compute_metrics,
    TensorBoardCallback,
    get_class_weights,
    CustomTrainer
)
from src.utils.utils import seed_everything, SEED, LACD_DATASET_PATH
import os
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="klue/roberta-base")
    parser.add_argument("--mode", type=str, help="train, test, or inference", default="train")
    
    # False, trainCivil, trainCriminal
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")

    parser.add_argument("--case_multiplier", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)

    parser.add_argument("--method", type=str, help="baseline or case-augmentation or case-concat-augmentation or rule-augmentation or rule-augmentation-hierarchical", default="baseline")
    parser.add_argument("--seed", type=int, help="seed", default=42)

    parser.add_argument("--fivefold", type=bool, help="fivefold", default=False)
    parser.add_argument("--fivefold_num", type=int, help="fivefold number", default=0)

    args = parser.parse_args()

    seed_everything(args.seed)

    case_multiplier = args.case_multiplier
    model_name = args.model
    tag = args.tag
    epoch_num = args.epoch


    model = CrossEncoderModel(model_name=model_name)
    tokenizer = model.tokenizer
    model.encoder.resize_token_embeddings(len(tokenizer))

    if args.fivefold:
        dataset_path = './data/datasets/LACD-biclassification/train-test-divide-fivefold/fold_' + str(args.fivefold_num) + '/'
    else:
        dataset_path = LACD_DATASET_PATH

    # 분할된 인덱스를 사용하여 train, test 데이터프레임 생성
    train_df = pd.read_json(dataset_path + 'train.jsonl', lines=True)
    test_df = pd.read_json(dataset_path + 'test.jsonl', lines=True)
    val_df = pd.read_json(dataset_path + 'val.jsonl', lines=True)

    original_train_df = train_df.copy()

    max_length = min([MAX_TOKEN_LENGTH, tokenizer.model_max_length])

    
    # if args.method == "rule-augmentation-hierarchical":
    train_dataset = NLIDataset(train_df, tokenizer, max_length=max_length, method=args.method)
    test_dataset = NLIDataset(test_df, tokenizer, max_length=max_length, method=args.method)
    val_dataset = NLIDataset(val_df, tokenizer, max_length=max_length, method=args.method)
    # else:
    #     train_dataset = NLIDataset(train_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)
    #     test_dataset = NLIDataset(test_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)
    #     val_dataset = NLIDataset(val_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)

    print(f"Token limit for {args.model}: {tokenizer.model_max_length}")
    train_dataset.print_label_counts()
    test_dataset.print_label_counts()
    val_dataset.print_label_counts()
    
    training_args = TrainingArguments(
        output_dir=f'./outputs/LACD-cross/small-fine-tune/{tag}',  # output directory
        num_train_epochs=epoch_num,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/LACD-cross/small-fine-tune/{tag}',  # directory for storing logs
        eval_strategy="epoch",  # evaluation strategy
        eval_steps=1,  # evaluation interval
        save_strategy="epoch",  # save strategy to match eval steps
        save_steps=1,  # save interval matching eval steps
        save_total_limit=1,  # only keep the best model
    )
    
    if args.mode == "train":
        print('train')

        writer = SummaryWriter()

        # 클래스 가중치 계산
        class_weights = get_class_weights(train_df)

        # 기존 Trainer 대신 CustomTrainer 사용
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(writer)],
            class_weights=class_weights,
        )

        trainer.train()

        test_results = trainer.evaluate(eval_dataset=test_dataset)

        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)

        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        writer.add_scalar("Test/F1", test_f1)
        writer.add_scalar("Test/Accuracy", test_accuracy)
        writer.add_scalar("Test/ROC_AUC", test_roc_auc)

        if args.model_save_path == "None":
            path = f"./data/models/LACD-cross/{tag}"
        else:
            path = args.model_save_path
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, os.path.join(path, "model.pth"))
        tokenizer.save_pretrained(path)
        
        # 실험 설정(config.json) 저장
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    elif args.mode == "temperature-calibration":
        print("temperature calibration")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(f"./data/models/LACD-cross/{tag}/model.pth")
        model.eval()
        
        # Initialize and optimize temperature parameter
        temperature = torch.nn.Parameter(torch.ones(1, device=device))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        # Collect logits from validation dataset
        all_logits = []
        all_labels = []
        
        # Collect logits using val_dataset
        with torch.no_grad():
            for i in range(len(val_dataset)):
                batch = val_dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0).to(device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
                labels = batch['labels'].unsqueeze(0).to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Temperature optimization function
        def eval():
            optimizer.zero_grad()
            scaled_logits = all_logits / temperature
            loss = torch.nn.BCEWithLogitsLoss()(scaled_logits, all_labels.float().unsqueeze(-1))
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        print(f"Optimized temperature parameter: {temperature.item()}")
        
        # Apply temperature parameter to the model
        class TemperatureScaledModel(torch.nn.Module):
            def __init__(self, model, temperature):
                super().__init__()
                self.model = model
                self.temperature = temperature
            
            def forward(self, input_ids=None, attention_mask=None, labels=None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Apply temperature scaling
                if hasattr(outputs, 'logits'):
                    outputs.logits = outputs.logits / self.temperature
                
                return outputs
        
        # Create temperature scaled model
        calibrated_model = TemperatureScaledModel(model, temperature.item())
        
        # Create model save path
        tempcal_path = f"./data/models/LACD-cross/{tag}/tempcal"
        if not os.path.exists(tempcal_path):
            os.makedirs(tempcal_path)
        
        # Save calibrated model
        torch.save(calibrated_model, os.path.join(tempcal_path, "model.pth"))
        
        # Save tokenizer and config to the same directory
        tokenizer.save_pretrained(tempcal_path)
        
        # Save configuration with temperature parameter
        config_dict = vars(args)
        config_dict["temperature"] = temperature.item()
        with open(os.path.join(tempcal_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
            
        print(f"Temperature calibrated model saved to {tempcal_path}")


        

    elif args.mode == "test":
        print("evaluate")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = torch.load(f"./data/models/LACD-cross/{tag}/model.pth")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        eval_result = trainer.evaluate()

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

        # 추가: 각 테스트 샘플의 loss와 logits의 분포를 레이블별로 히스토그램으로 그리고 PNG 파일로 저장
        import os
        import torch
        from torch.utils.data import DataLoader
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # ./visualization/ 폴더가 없으면 생성
        os.makedirs("./visualization/", exist_ok=True)

        # 테스트 데이터셋에 대한 DataLoader 생성 (배치 사이즈는 상황에 맞게 조정)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        model.eval()
        all_losses_true = []
        all_losses_false = []
        all_logits_true = []
        all_logits_false = []

        # BCEWithLogitsLoss 사용 (reduction='none'로 각 샘플별 loss 계산)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        with torch.no_grad():
            for batch in test_loader:
                # 수정된 입력 키 반영: article1_idx, article2_idx, input_ids, attention_mask, labels
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # BCEWithLogitsLoss는 target이 float여야 함
                labels = batch["labels"].to(device).float()

                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                # outputs.logits의 shape가 (batch_size, 1)인 경우 squeeze()로 (batch_size,)로 변환
                logits = outputs.logits.squeeze()
                loss = criterion(logits, labels)

                # 레이블별로 분리: True (1.0)와 False (0.0)
                mask_true = (labels == 1.0)
                mask_false = (labels == 0.0)

                if mask_true.any():
                    all_losses_true.extend(loss[mask_true].detach().cpu().numpy())
                    all_logits_true.extend(logits[mask_true].detach().cpu().numpy())
                if mask_false.any():
                    all_losses_false.extend(loss[mask_false].detach().cpu().numpy())
                    all_logits_false.extend(logits[mask_false].detach().cpu().numpy())

        # 히스토그램 그리기: 첫 번째 데이터셋(True)이 아래쪽에 쌓이도록 순서를 [True, False]로 전달
        plt.figure()
        plt.hist([all_losses_true, all_losses_false], bins=50, stacked=True, color=['red', 'blue'], label=['True', 'False'])
        plt.title("Test Loss Histogram")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("./visualization/test_loss_histogram_baseline.png")
        plt.close()

        plt.figure()
        plt.hist([all_logits_true, all_logits_false], bins=50, stacked=True, color=['red', 'blue'], label=['True', 'False'])
        plt.title("Test Logits Histogram")
        plt.xlabel("Logits")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("./visualization/test_logits_histogram_baseline.png")
        plt.close()

        # 추가: True와 False 레이블에 해당하는 정확한 logit 값을 CSV 파일로 저장
        df_true = pd.DataFrame({
            "label": [True] * len(all_logits_true),
            "logit": all_logits_true
        })
        df_false = pd.DataFrame({
            "label": [False] * len(all_logits_false),
            "logit": all_logits_false
        })
        # 두 DataFrame을 하나로 합치기
        df_logits = pd.concat([df_true, df_false], ignore_index=True)
        df_logits.to_csv("./visualization/test_logits_baseline.csv", index=False)

