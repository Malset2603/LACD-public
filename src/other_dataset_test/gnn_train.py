
import torch
from torch.utils.tensorboard import SummaryWriter # type: ignore
from transformers import Trainer, TrainingArguments
import numpy as np
import argparse
from chromadb import PersistentClient
from tqdm import trange, tqdm
import json
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.other_dataset_test.gnn_models import  NoGNNCrossEncoderModel, GATv2CrossEncoderModel
from src.utils.utils import article_key_function, seed_everything, SEED
import pandas as pd
from transformers import AutoTokenizer, AutoModel



from src.utils.encoder.utils import (
    MAX_TOKEN_LENGTH,
    compute_metrics,
    TensorBoardCallback,
    get_class_weights,
    CustomTrainer
)
from src.other_dataset_test.dataset import GNNNLIDataset
import os

def load_dataset(jsonl_file, article_network, tokenizer, batch_size=128):
    import json
    from tqdm import tqdm
    from transformers import AutoTokenizer

    # 준비: query encoder tokenizer와 최대 토큰 길이 설정
    query_encoder_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", clean_up_tokenization_spaces=True)
    max_length = min(MAX_TOKEN_LENGTH, tokenizer.model_max_length)
    query_max_length = min(MAX_TOKEN_LENGTH, query_encoder_tokenizer.model_max_length)

    dataset = []
    key_error_cnt = 0

    # 임시 배치 리스트 초기화
    batch_queries = []
    batch_articles = []
    batch_article_idxs = []
    batch_answers = []

    def process_batch():
        """배치 내 데이터들을 토큰화하고 dataset에 추가한 후, 배치를 초기화한다."""
        nonlocal batch_queries, batch_articles, batch_article_idxs, batch_answers, dataset

        # query와 article 쌍 배치 인코딩
        encoding = tokenizer.batch_encode_plus(
            list(zip(batch_queries, batch_articles)),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # query만 별도 배치 인코딩
        query_encoding = query_encoder_tokenizer.batch_encode_plus(
            batch_queries,
            add_special_tokens=True,
            max_length=query_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # 각 배치 샘플을 dataset에 추가
        for i in range(len(batch_queries)):
            dataset.append((
                batch_article_idxs[i],
                query_encoding['input_ids'][i],
                query_encoding['attention_mask'][i],
                encoding['input_ids'][i],
                encoding['attention_mask'][i],
                batch_answers[i]
            ))
        # 배치 초기화
        batch_queries.clear()
        batch_articles.clear()
        batch_article_idxs.clear()
        batch_answers.clear()

    # JSONL 파일 읽으며 배치 처리
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            query = data['query']
            article = data['article']
            try:
                article_idx = article_network.article_key_to_idx[article_key_function(article)]
            except KeyError:
                key_error_cnt += 1
                continue

            batch_queries.append(query)
            batch_articles.append(article)
            batch_article_idxs.append(article_idx)
            batch_answers.append(1 if data['answer'] else 0)

            if len(batch_queries) >= batch_size:
                process_batch()

        # 남은 데이터 처리
        if batch_queries:
            process_batch()

    return dataset


# Example execution
if __name__ == "__main__":

    DATASET_PATH = "/mnt/disk1/anseon2001/LACD/data/database/lbox_open_statute_classification/"


    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train or test", default="train")
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base")
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--chroma_db_name", type=str, required=True, help="Name of the Chroma DB where encodings will be stored")
    parser.add_argument("--gnn_method", type=str, help="Name of GNN method", default="gcn")
    parser.add_argument("--epoch", type = int, default=1)

    # parser.add_argument("--no_cross", type=bool, default=False)
    parser.add_argument("--gnn_edge_way", type=str, choices=["both", "forward", "backward"], help="The way for edges in CAMGraph", default="both")
    parser.add_argument("--seed",type=int,help="seed",default=42)



    args = parser.parse_args()

    seed_everything(args.seed)

    chroma_db_name = args.chroma_db_name
    model_name = args.model
    gnn_method = args.gnn_method
    epoch_num = args.epoch


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if not os.path.exists("./data/database/chroma_db/" + chroma_db_name):
        print("Chroma DB does not exist")
        assert(0)

    # Initialize Chroma DB client before creating the model
    client = PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart")

    # Article Network 만들기
    article_network = ArticleNetwork(edge_way=args.gnn_edge_way)
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)

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

    # vectors 만들기. tensor 로 만들어야 함.
    max_node_idx = edge_index_tensor.max().item()
    vectors_list = []
    for idx in range(int(max_node_idx + 1)):
        if idx in collection_dict.keys():
            vectors_list.append(torch.tensor(collection_dict[idx]["embedding"], dtype=torch.float32).to(device))
        else:
            vectors_list.append(torch.zeros(embedding_size, dtype=torch.float32).to(device))
    vector_tensor = torch.stack(vectors_list).to(device)

    # GNN Bi-Encoder 모델 초기화 using dynamic embedding size
    if gnn_method == "gat":
        model = GATv2CrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    elif gnn_method == "vanilla":
        model = NoGNNCrossEncoderModel(model_name=model_name, in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    else:
        print("improper GNN methods!")
        assert(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tokenizer = model.tokenizer
    model.encoder.resize_token_embeddings(len(tokenizer))


    # Load datasets

    train_dataset = GNNNLIDataset(load_dataset(DATASET_PATH+'train.jsonl', article_network, tokenizer))
    # we do not need to multiply cases for val, test datasets
    val_dataset = GNNNLIDataset(load_dataset(DATASET_PATH+'val.jsonl', article_network, tokenizer))
    test_dataset = GNNNLIDataset(load_dataset(DATASET_PATH+'test.jsonl', article_network, tokenizer))




    training_args = TrainingArguments(
        output_dir=f'../outputs/lbox/small-fine-tune/gnns/{args.tag}',  # output directory
        num_train_epochs=epoch_num,                          # total number of training epochs
        per_device_train_batch_size=8,               # batch size for training
        per_device_eval_batch_size=8,                # batch size for evaluation
        warmup_steps=500,                            # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/lbox/small-fine-tune/gnns/{args.tag}',  # directory for storing logs
        eval_strategy="epoch",                       # evaluation strategy
        eval_steps=1,                               # evaluation interval
        save_strategy="epoch",                       # save strategy to match eval steps
        save_steps=1,                               # save interval matching eval steps
        save_total_limit=1,                          # only keep the best model
        # load_best_model_at_end=True,                 # load the best model at the end
        # metric_for_best_model="eval_loss",           # metric to use for model selection
        report_to="tensorboard"                      # report to TensorBoard
    )

    writer = SummaryWriter()

    train_df = pd.read_json(DATASET_PATH + 'train.jsonl', lines=True)
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
            path = f"./data/models/lbox/gnns/{args.tag}"
        else:
            path = args.model_save_path
        
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, path+"/model.pth")
        tokenizer.save_pretrained(path)


    elif args.mode == "test":
        print("evaluate")
        model = torch.load(f"./data/models/lbox/gnns/{args.tag}/model.pth")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
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
    
