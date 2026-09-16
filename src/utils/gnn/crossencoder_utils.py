import torch
from torch.utils.data import Dataset

class GNNNLIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_weights = self.get_class_weights()  # 데이터셋 로딩 시 자동으로 가중치 계산

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article1_idx, article2_idx, input_ids, attention_mask, label = self.dataset[idx]
        return {
            'article1_idx': torch.tensor(article1_idx, dtype=torch.long),
            'article2_idx': torch.tensor(article2_idx, dtype=torch.long),
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
