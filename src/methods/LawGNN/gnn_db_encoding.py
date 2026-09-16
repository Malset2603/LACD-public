
import chromadb
import torch
import numpy as np

from tqdm import tqdm

from src.utils.encoder.utils import MAX_TOKEN_LENGTH
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.utils.utils import article_key_function

from src.methods.LawGNN.gnn_architecture import GCNBiEncoder, SAGEBiEncoder, GATv2BiEncoder

import argparse


# 이미 저장된 DB 에 GNN layer 를 얹는 용도이다.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB encoder by trained GNN models")


    parser.add_argument("--gnn_model_path", type=str, help="GNN model path", default="./data/models/LACD-bi/gnns/gcn")
    parser.add_argument("--chroma_db_name", type=str, required=True, help="Name of the Chroma DB where encodings will be stored")
    parser.add_argument("--method", type=str, help="cosine or linear", default="cosine")
    parser.add_argument("--gnn_method", type=str, help="Name of GNN method", default = "gcn")


    args = parser.parse_args()

    # 사용 예시
    gnn_model_path = args.gnn_model_path
    chroma_db_name = args.chroma_db_name
    biencoder_classification_method = args.method
    gnn_method = args.gnn_method


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    client = chromadb.PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart")

    new_db_name = chroma_db_name + "_"+gnn_method
    new_client = chromadb.PersistentClient(path="./data/database/chroma_db/" + new_db_name)
    new_chroma_collection = new_client.get_or_create_collection("quickstart")

    # Article Network 만들기
    article_network = ArticleNetwork()
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)


    # Load all contents from Chroma DB
    all_contents = chroma_collection.get(include=["embeddings", "documents"])
    embeddings = all_contents.get("embeddings", [])
    documents = all_contents.get("documents", [])


    if len(embeddings) > 0:
        embedding_size = len(embeddings[0])
    else:
        raise ValueError("No embeddings found in ChromaDB.")
    
    # print("1. embedding_size is", embedding_size)

    
    if gnn_method == "gcn":
        model = GCNBiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=biencoder_classification_method).to(device)
    elif gnn_method == "graphsage":
        model = SAGEBiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=biencoder_classification_method).to(device)
    elif gnn_method == "gat":
        model = GATv2BiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=biencoder_classification_method).to(device)
    else:
        print("improper GNN methods!")
        assert(0)    
    model.load_state_dict(torch.load(gnn_model_path + "/model.pth"))
    model.eval()  # 모델을 평가 모드로 설정

    collection_dict = {}
    no_key_count = 0
    for i in range(len(embeddings)):  # Use documents length as the reference

        article_key = article_key_function(documents[i])
        try:
            article_idx = article_network.article_key_to_idx[article_key]
        except:
            print(article_key)
            no_key_count = no_key_count + 1
            exit(0)

            
        entry = {
            "embedding": np.array(embeddings[i]) if i < len(embeddings) else None,
            "document": documents[i],
            "article_key": article_key
        }
        collection_dict[article_idx] = entry


    max_node_idx = edge_index_tensor.max().item()

    vectors_list = []
    for idx in range(int(max_node_idx + 1)):
        if idx in collection_dict.keys():
            vectors_list.append(torch.tensor(collection_dict[idx]["embedding"], dtype=torch.float32).to(device))
        else:
            vectors_list.append(torch.zeros(embedding_size, dtype=torch.float32).to(device))

    vector_tensor = torch.stack(vectors_list).to(device)
    # print(vector_tensor.shape)

    # model apply
    vector_tensor = model.encode(vector_tensor, edge_index_tensor)

    # vector tensor 삽입
    # TODO: 다음 코드를 참고하여 new_chroma_db 에 vector_tensor 안의 embedding 을 저장하기

    # 0인 vector 는 넣을 필요 없음.
    # Preparing data for storage in ChromaDB
    chunk_size = 1024
    total_chunks = len(vector_tensor) // chunk_size + 1
    vector_embeddings = vector_tensor.cpu().detach().numpy()  # Move tensor to CPU and convert to numpy
    vector_embeddings = [e.flatten().tolist() for e in vector_embeddings]  # Convert embeddings to lists

    filtered_embeddings = []
    filtered_documents = []
    filtered_ids = []
    skip_count = 0
    # Filter out zero vectors
    for idx in tqdm(range(len(vector_embeddings))):
        if not np.all(np.isclose(vector_embeddings[idx], 0)) and idx in collection_dict.keys():  # Skip if embedding is all zeros

            filtered_embeddings.append(vector_embeddings[idx])
            filtered_documents.append(collection_dict[idx]["document"])
            filtered_ids.append(str(idx))

        else:
            skip_count = skip_count+1
    # print("2. embedding size is ", len(vector_embeddings[idx]))
    print("skip count: {}".format(skip_count))
    print("saving list length: {}".format(len(filtered_ids)))

    # Insert the filtered embeddings into the new ChromaDB collection
    chunk_size = 1024
    total_chunks = len(filtered_embeddings) // chunk_size + 1

    for chunk_idx in tqdm(range(total_chunks)):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(filtered_embeddings))

        # Extract chunk of embeddings, documents, and IDs
        chunk_embeddings = filtered_embeddings[start_idx:end_idx]
        chunk_ids = filtered_ids[start_idx:end_idx]
        chunk_docs = filtered_documents[start_idx:end_idx]

        # Add chunk to the new Chroma collection
        new_chroma_collection.add(
            documents=chunk_docs,
            embeddings=chunk_embeddings,
            ids=chunk_ids,
        )