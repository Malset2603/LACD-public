
import torch
from torch.utils.tensorboard import SummaryWriter # type: ignore
from transformers import Trainer, TrainingArguments
import numpy as np
import argparse
from chromadb import PersistentClient
from tqdm import trange
import json
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.methods.LawGNN.gnn_architecture import GCNCrossEncoderModel, NoGNNCrossEncoderModel, SAGECrossEncoderModel, GATv2CrossEncoderModel
from src.utils.utils import article_key_function, seed_everything, SEED, LACD_DATASET_PATH
import polars as pl
import copy

from src.utils.encoder.utils import (
    MAX_TOKEN_LENGTH,
    compute_metrics,
    TensorBoardCallback,
    get_class_weights,
    CustomTrainer
)
from src.utils.gnn.crossencoder_utils import GNNNLIDataset
import os

# Function to load data from jsonl — early-load
def load_dataset(jsonl_file, article_network, tokenizer, method="baseline", case_multiplier=1, subset_ratio=None, subset_seed=42, mini_ratio=None, mini_seed=None):
    dataset = []
    key_error_cnt = 0
    # early-load: sampling while reading the file, do not load 100% then slice (alias mini_* deprecated)
    if subset_ratio is None and mini_ratio is not None:
        subset_ratio = mini_ratio
    if subset_seed == 42 and mini_seed is not None:
        subset_seed = mini_seed
    if subset_ratio is not None and subset_ratio < 1.0:
        from src.utils.utils import load_jsonl_early
        all_lines, orig = load_jsonl_early(jsonl_file, ratio=subset_ratio, seed=subset_seed, label_key="answer")
        print(f"[SUBSET] early-load {jsonl_file} {orig}->{len(all_lines)} ratio={subset_ratio} pos {sum(1 for r in all_lines if r.get('answer'))}/{len(all_lines)}")
    else:
        all_lines = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_lines.append(json.loads(line))

    for c_m in range(case_multiplier):
        for data in all_lines:
                article1 = data['article1']
                article2 = data['article2']
                try:
                    article1_idx = article_network.article_key_to_idx[article_key_function(article1)]
                    article2_idx = article_network.article_key_to_idx[article_key_function(article2)]
                except:
                    key_error_cnt += 1
                    continue

                premise = article1
                hypothesis = article2

                max_length = min([MAX_TOKEN_LENGTH, tokenizer.model_max_length])

                encoding = tokenizer.encode_plus(
                    text= premise,
                    text_pair=hypothesis,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                label = 1 if data['answer'] else 0  # True -> 1, False -> 0
                dataset.append((article1_idx, article2_idx, encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), label))

            

            
    return dataset

# Example execution
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train or test", default="train")
    parser.add_argument("--model", type=str, default="klue/roberta-base")
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--chroma_db_name", type=str, required=True, help="Name of the Chroma DB where encodings will be stored")
    parser.add_argument("--gnn_method", type=str, help="Name of GNN method", default="gcn")
    parser.add_argument("--case_augmentation_method", type=str, help="case augmentation method. case-augmentation or baseline or rule-augmentation", default="baseline")
    parser.add_argument("--epoch", type = int, default=5)

    parser.add_argument("--case_multiplier", type=int, default=1)
    # parser.add_argument("--no_cross", type=bool, default=False)
    parser.add_argument("--gnn_edge_way", type=str, choices=["both", "forward", "backward"], help="The way for edges in LMGraph", default="both")
    parser.add_argument("--seed",type=int,help="seed",default=42)

    parser.add_argument("--gnn_append_mode",type=str, choices=["baseline", "append"],default="baseline")
    parser.add_argument("--gnn_append_model_tag",type=str, default="roberta-0")

    # fivefold
    parser.add_argument("--fivefold", type=bool, default=False)
    parser.add_argument("--fivefold_num", type=int, default=0)


    parser.add_argument("--subset_ratio", "--sample_ratio", "--mini_ratio", type=float, default=None, dest="subset_ratio", help="subset sampling: fraction (0,1] stratified sampling, e.g. 0.15 (alias --mini_ratio deprecated)")
    parser.add_argument("--subset_seed", "--sample_seed", "--mini_seed", type=int, default=42, dest="subset_seed", help="seed for subset sampling (alias --mini_seed deprecated)")
    parser.add_argument("--subset_laws", "--mini_laws", "--law_nodes", type=int, default=None, dest="subset_laws", help="subset laws: limit number of articles (graph-aware) (alias --mini_laws deprecated)")
    parser.add_argument("--max_length", type=int, default=4096, help="max token length (default 4096, use 512 for mini VRAM)")
    parser.add_argument("--fp16", action="store_true", help="enable fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")

    # optional metrics output (no default; if not provided, metrics are only printed)
    parser.add_argument("--metrics_output", type=str, default=None, help="optional path to save metrics JSON (e.g. ./outputs/metrics/cross-10k.json)")

    args = parser.parse_args()
    # backward compat aliases
    args.mini_ratio = args.subset_ratio
    args.mini_laws = args.subset_laws
    args.mini_seed = args.subset_seed

    seed_everything(args.seed)

    chroma_db_name = args.chroma_db_name
    model_name = args.model
    gnn_method = args.gnn_method
    case_augmentation_method = args.case_augmentation_method
    case_multiplier = args.case_multiplier
    epoch_num = args.epoch


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("./data/database/chroma_db/" + chroma_db_name):
        print("Chroma DB does not exist")
        assert(0)

    # Initialize Chroma DB client before creating the model
    client = PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart")

    # Create Article Network (edge_way and subset_laws support)
    article_network = ArticleNetwork(edge_way=args.gnn_edge_way, subset_laws=args.subset_laws, subset_seed=args.subset_seed)
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)
    if args.subset_laws is not None:
        print(f"[SUBSET] ArticleNetwork nodes {len(article_network.all_article_keys)} (subset_laws={args.subset_laws}) edges {edge_index_tensor.shape[1]//2} (bidirectional)")

    # propagate max_length/fp16 to utils if provided
    if args.max_length != 4096:
        from src.utils.encoder import utils as _enc_utils
        _enc_utils.MAX_TOKEN_LENGTH = args.max_length
        print(f"[SUBSET] MAX_TOKEN_LENGTH overridden -> {args.max_length}")
    if args.gradient_checkpointing:
        print(f"[SUBSET] gradient_checkpointing requested (GNN LM part only if supported)")

    # Load all contents from Chroma DB
    all_contents = chroma_collection.get(include=["embeddings", "documents"]) # type: ignore
    embeddings = all_contents.get("embeddings", [])
    documents = all_contents.get("documents", [])

    if len(embeddings) > 0: # type: ignore
        embedding_size = len(embeddings[0]) # type: ignore
    else:
        raise ValueError("No embeddings found in ChromaDB.")
    
    collection_dict = {}
    no_key_count = 0
    for i in range(len(embeddings)):  # type: ignore # Use documents length as the reference
        article_key = article_key_function(documents[i]) # type: ignore
        try:
            article_idx = article_network.article_key_to_idx[article_key]
        except:
            no_key_count += 1
            continue  # Skip if key not found
        entry = {
            "embedding": np.array(embeddings[i]) if i < len(embeddings) else None, # type: ignore
            "article_key": article_key
        }
        collection_dict[article_idx] = entry

    # Model initialization

    # Build vector table sized to total nodes (not max edge index) to include isolated nodes.
    num_nodes = len(article_network.all_article_keys)
    vectors_list = []
    for idx in range(num_nodes):
        if idx in collection_dict.keys():
            vectors_list.append(torch.tensor(collection_dict[idx]["embedding"], dtype=torch.float32).to(device))
        else:
            vectors_list.append(torch.zeros(embedding_size, dtype=torch.float32).to(device))
    vector_tensor = torch.stack(vectors_list).to(device)

    # Initialize GNN Bi-Encoder model using dynamic embedding size
    if gnn_method == "gcn":
        model = GCNCrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    elif gnn_method == "graphsage":
        model = SAGECrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    elif gnn_method == "gat":
        model = GATv2CrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    elif gnn_method == "vanilla":
        model = NoGNNCrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    else:
        # print("improper GNN methods!")
        assert(0)


    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tokenizer = model.tokenizer
    model.encoder.resize_token_embeddings(len(tokenizer))

    batch_size = 16

    if args.gnn_append_mode == "append":
        original_model = torch.load(f"./data/models/LACD-cross/{args.gnn_append_model_tag}/model.pth")
        model.encoder = copy.deepcopy(original_model.encoder)
        for param in model.encoder.parameters():
            param.requires_grad = False

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = count_trainable_parameters(model)
    print(f"Number of trainable parameters: {trainable_params}")

    if args.fivefold:
        dataset_path = './data/datasets/LACD-biclassification/train-test-divide-fivefold/fold_' + str(args.fivefold_num) + '/'
    else:
        dataset_path = LACD_DATASET_PATH
    # Load datasets (subset_ratio forwarded, alias mini_ratio)
    train_dataset = GNNNLIDataset(load_dataset(dataset_path + 'train.jsonl', article_network, tokenizer, method=case_augmentation_method, case_multiplier=case_multiplier, subset_ratio=args.subset_ratio, subset_seed=args.subset_seed))
    # we do not need to multiply cases for val, test datasets
    val_dataset = GNNNLIDataset(load_dataset(dataset_path + 'val.jsonl', article_network, tokenizer, method=case_augmentation_method, subset_ratio=args.subset_ratio, subset_seed=args.subset_seed))
    test_dataset = GNNNLIDataset(load_dataset(dataset_path + 'test.jsonl', article_network, tokenizer, method=case_augmentation_method, subset_ratio=args.subset_ratio, subset_seed=args.subset_seed))

    training_args = TrainingArguments(
        output_dir=f'./outputs/LACD-cross/gnns/{args.tag}',  # output directory
        num_train_epochs=epoch_num,                          # total number of training epochs
        per_device_train_batch_size=batch_size,               # batch size for training
        per_device_eval_batch_size=batch_size,                # batch size for evaluation
        warmup_steps=500,                            # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/LACD-cross/gnns/{args.tag}',  # directory for storing logs
        eval_strategy="epoch",                       # evaluation strategy
        eval_steps=1,                               # evaluation interval
        save_strategy="epoch",                       # save strategy to match eval steps
        save_steps=1,                               # save interval matching eval steps
        save_total_limit=1,                          # only keep the best model
        report_to="tensorboard",                      # report to TensorBoard

        # load_best_model_at_end=True,                 # load the best model at the end
        # metric_for_best_model="eval_loss",           # metric to use for model selection
    )

    writer = SummaryWriter()

    train_df = pd.read_json(dataset_path + 'train.jsonl', lines=True)
    class_weights = get_class_weights(train_df)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(writer)],
        class_weights=class_weights,
    )


    if args.mode == "train":
        print("Training...")
        trainer.train()
        test_results = trainer.evaluate(eval_dataset=test_dataset)
         
        # Save results
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)

        # Print results
        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        # Record to TensorBoard
        writer.add_scalar("Test/F1", test_f1)
        writer.add_scalar("Test/Accuracy", test_accuracy)
        writer.add_scalar("Test/ROC_AUC", test_roc_auc)

        # optionally save metrics to file
        if args.metrics_output:
            import json as _json
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)) or ".", exist_ok=True)
            _payload = {
                "tag": args.tag,
                "mode": args.mode,
                "gnn_method": gnn_method,
                "metrics": {"f1": test_f1, "accuracy": test_accuracy, "roc_auc": test_roc_auc, "precision": test_results.get("eval_precision", 0), "recall": test_results.get("eval_recall", 0)},
                "args": vars(args),
            }
            with open(args.metrics_output, 'w', encoding='utf-8') as _f:
                _json.dump(_payload, _f, ensure_ascii=False, indent=2)
            print(f"[METRICS] saved to {args.metrics_output}")

        if args.model_save_path == "None":
            path = f"./data/models/LACD-cross/gnns/{args.tag}"
        else:
            path = args.model_save_path
        
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, path+"/model.pth")
        tokenizer.save_pretrained(path)

    elif args.mode == "temperature-calibration":
        print("temperature calibration")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(f"./data/models/LACD-cross/gnns/{args.tag}/model.pth")
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
                # Each value is already 1D tensor
                article1_idx = batch['article1_idx'].unsqueeze(0).to(device)
                article2_idx = batch['article2_idx'].unsqueeze(0).to(device)
                input_ids = batch['input_ids'].unsqueeze(0).to(device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
                labels = batch['labels'].unsqueeze(0).to(device)

                outputs = model(
                    article1_idx=article1_idx,
                    article2_idx=article2_idx,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
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
            
            def forward(self, x, edge_index, a_idx, b_idx, labels=None):
                outputs = self.model(x, edge_index, a_idx, b_idx, labels=labels)
                
                # Apply temperature scaling
                if hasattr(outputs, 'logits'):
                    outputs.logits = outputs.logits / self.temperature
                
                return outputs
        
        # Create temperature scaled model
        calibrated_model = TemperatureScaledModel(model, temperature.item())
        
        # Create model save path
        tempcal_path = f"./data/models/LACD-cross/gnns/{args.tag}/tempcal"
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
        model = torch.load(f"./data/models/LACD-cross/gnns/{args.tag}/model.pth")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            compute_metrics=compute_metrics
        )
        
        test_results = trainer.evaluate()

        # Print test results
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)
        test_precision = test_results.get("eval_recall", 0)
        test_recall = test_results.get("eval_precision", 0)

        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test precision Score: {test_precision:.1%}")
        print(f"Test recall Score: {test_recall:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        if args.metrics_output:
            import json as _json
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)) or ".", exist_ok=True)
            _payload = {
                "tag": args.tag,
                "mode": args.mode,
                "gnn_method": gnn_method,
                "metrics": {"f1": test_f1, "accuracy": test_accuracy, "roc_auc": test_roc_auc, "precision": test_precision, "recall": test_recall},
                "args": vars(args),
            }
            with open(args.metrics_output, 'w', encoding='utf-8') as _f:
                _json.dump(_payload, _f, ensure_ascii=False, indent=2)
            print(f"[METRICS] saved to {args.metrics_output}")

        # Additional: Draw histograms of loss and logits distribution by label and save as PNG files
        import os
        import torch
        from torch.utils.data import DataLoader
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Create ./visualization/ directory if it doesn't exist
        os.makedirs("./visualization/", exist_ok=True)

        # Create DataLoader for test dataset (adjust batch size as needed)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        model.eval()
        all_losses_true = []
        all_losses_false = []
        all_logits_true = []
        all_logits_false = []

        # Use BCEWithLogitsLoss (with reduction='none' to calculate loss per sample)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        with torch.no_grad():
            for batch in test_loader:
                # Modified input keys: article1_idx, article2_idx, input_ids, attention_mask, labels
                article1_idx = batch["article1_idx"].to(device)
                article2_idx = batch["article2_idx"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # BCEWithLogitsLoss requires float target
                labels = batch["labels"].to(device).float()

                outputs = model(
                    article1_idx=article1_idx, 
                    article2_idx=article2_idx,
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                # Squeeze outputs.logits from (batch_size, 1) to (batch_size,)
                logits = outputs.logits.squeeze()
                loss = criterion(logits, labels)

                # Separate by label: True (1.0) and False (0.0)
                mask_true = (labels == 1.0)
                mask_false = (labels == 0.0)

                if mask_true.any():
                    all_losses_true.extend(loss[mask_true].detach().cpu().numpy())
                    all_logits_true.extend(logits[mask_true].detach().cpu().numpy())
                if mask_false.any():
                    all_losses_false.extend(loss[mask_false].detach().cpu().numpy())
                    all_logits_false.extend(logits[mask_false].detach().cpu().numpy())

        # Draw histogram: Pass [True, False] order to stack True dataset at bottom
        plt.figure()
        plt.hist([all_losses_true, all_losses_false], bins=50, stacked=True, color=['red', 'blue'], label=['True', 'False'])
        plt.title("Test Loss Histogram")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("./visualization/test_loss_histogram.png")
        plt.close()

        plt.figure()
        plt.hist([all_logits_true, all_logits_false], bins=50, stacked=True, color=['red', 'blue'], label=['True', 'False'])
        plt.title("Test Logits Histogram")
        plt.xlabel("Logits")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("./visualization/test_logits_histogram.png")
        plt.close()

        # Additional: Save exact logit values for True and False labels to CSV
        df_true = pl.DataFrame({
            "label": [True] * len(all_logits_true),
            "logit": all_logits_true
        })
        df_false = pl.DataFrame({
            "label": [False] * len(all_logits_false),
            "logit": all_logits_false
        })
        # Concatenate two DataFrames
        df_logits = pl.concat([df_true, df_false], how="vertical")
        df_logits.write_csv("./visualization/test_logits.csv")
