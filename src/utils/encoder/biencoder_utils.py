import os
import json
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name, method='cosine'):
        super(BiEncoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method  # Choose between 'cosine' or 'linear'

        if self.method == 'linear':
            hidden_size = self.encoder.config.hidden_size
            # Linear layer for classification
            self.classifier = torch.nn.Linear(hidden_size * 2, 1)  # Binary classification

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        labels=None,
    ):
        output_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        output_b = self.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b)

        # Use the [CLS] token representations
        pooled_output_a = output_a.last_hidden_state[:, 0, :]
        pooled_output_b = output_b.last_hidden_state[:, 0, :]

        pooled_output_a = self.dropout(pooled_output_a)
        pooled_output_b = self.dropout(pooled_output_b)

        if self.method == 'cosine':
            # 내적(inner product) 계산
            inner_prod = torch.sum(pooled_output_a * pooled_output_b, dim=1)
            logits = inner_prod.unsqueeze(-1)  # Shape [batch_size, 1]

        elif self.method == 'linear':
            # Concatenate the embeddings and use a linear layer for classification
            combined_output = torch.cat([pooled_output_a, pooled_output_b], dim=1)
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))  # Ensure labels match logits shape

        return SequenceClassifierOutput(loss=loss, logits=logits)


class DPRBiEncoderModel(torch.nn.Module):
    """
    Dense Passage Retrieval (DPR) style bi-encoder.
    Produces dense embeddings for queries and passages and computes
    dot-product scores for retrieval.
    """
    def __init__(self, model_name, dropout=0.1):
        super(DPRBiEncoderModel, self).__init__()
        # Separate encoders for query and passage
        self.encoder = AutoModel.from_pretrained(model_name)
        self.passage_encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)

        self.passage_encoder.eval()
        # Optionally tie weights:
        # self.passage_encoder = self.question_encoder

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        p_input_ids,
        p_attention_mask,
        labels=None
    ):
        # Encode queries
        q_outputs = self.encoder(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            return_dict=True
        )
        q_emb = q_outputs.last_hidden_state[:, 0, :]  # [CLS]
        q_emb = self.dropout(q_emb)

        # Encode passages (can be positives + negatives)
        # Expect shape [batch_size, num_ctx, seq_len]
        bsz, num_ctx, seq_len = p_input_ids.size()
        flat_p_ids = p_input_ids.view(-1, seq_len)
        flat_p_mask = p_attention_mask.view(-1, seq_len)

        p_outputs = self.passage_encoder(
            input_ids=flat_p_ids,
            attention_mask=flat_p_mask,
            return_dict=True
        )
        p_emb = p_outputs.last_hidden_state[:, 0, :]
        p_emb = self.dropout(p_emb)

        # Restore [batch_size, num_ctx, hidden]
        p_emb = p_emb.view(bsz, num_ctx, -1)

        # Compute dot-product scores: [batch_size, num_ctx]
        # q_emb: [batch_size, hidden]
        # p_emb: [batch_size, num_ctx, hidden]
        scores = torch.einsum('bh,bnh->bn', q_emb, p_emb)

        loss = None
        if labels is not None:
            # labels: tensor of shape [batch_size], with idx of positive in each context set
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(scores, labels)

        return {
            'loss': loss,
            'scores': scores,
            'q_emb': q_emb,
            'p_emb': p_emb
        }

    def get_question_embedding(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = outputs.last_hidden_state[:, 0, :]
        return self.dropout(emb)

    def get_passage_embedding(self, input_ids, attention_mask):
        outputs = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = outputs.last_hidden_state[:, 0, :]
        return self.dropout(emb)


# Chroma DB utils

def load_chromaDB_byname(chroma_db_name):
    import chromadb
    # ChromaDB 클라이언트 초기화 및 인코딩된 법률 저장 또는 불러오기
    client = chromadb.PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart", metadata={"hnsw:space": "ip"})
    return chroma_collection
