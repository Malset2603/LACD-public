

import torch
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer


from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel

from src.utils.utils import article_key_function
from src.utils.encoder.utils import ThreeLayerClassifier


# lbox open 의 statute classification 을 위한 NLI dataset 
class NLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, method="baseline"):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.method = method

        self.label_map = {True: 1, False: 0}

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        premise = self.dataframe.iloc[index]["query"]
        hypothesis = self.dataframe.iloc[index]["article"]
        label = 1 if self.dataframe.iloc[index]["answer"] else 0

        if self.method == "without-article":
            hypothesis = article_key_function(hypothesis)

        elif self.method == "article-only":
            pass
        

        
            

        # 🚀 Baseline 및 기타 메서드의 경우 기존 방식 유지
        encoding = self.tokenizer.encode_plus(
            text=premise,
            text_pair=hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def print_label_counts(self):
        true_count = (self.dataframe["answer"] == True).sum()
        false_count = (self.dataframe["answer"] == False).sum()
        print(f"True labels: {true_count}")
        print(f"False labels: {false_count}")



class GNNNLIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_weights = self.get_class_weights()  # 데이터셋 로딩 시 자동으로 가중치 계산

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article_idx, query_input_ids, query_attention_mask, input_ids, attention_mask, label = self.dataset[idx]
        return {
            'article_idx': torch.tensor(article_idx, dtype=torch.long),
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }
    
    def get_class_weights(self):
        """
        데이터셋에서 각 클래스(0, 1)의 개수를 세고 가중치를 계산하는 함수
        """
        labels = [sample[-1] for sample in self.dataset]  # 마지막 값이 label
        num_samples = len(labels)
        num_classes = 2  # Binary Classification (0, 1)

        class_counts = torch.tensor([labels.count(0), labels.count(1)], dtype=torch.float)
        class_weights = num_samples / (num_classes * class_counts)

        return class_weights