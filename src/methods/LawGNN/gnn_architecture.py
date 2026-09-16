import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


# GCN architecture for Bi-Encoder
class GCNBiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, method='ip'):
        super(GCNBiEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method

        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'ip':
            # 내적 계산 (inner product)
            ip_sim = torch.sum(x[a_idx] * x[b_idx], dim=1)
            # 정규화된 값으로 변환
            ip_prob = torch.sigmoid(ip_sim)  # 내적 값을 [0, 1] 범위로 매핑
            logits = ip_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'cosine':
                # 내적 기반 손실 함수
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(ip_sim, labels.float())
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# GraphSAGE architecture for Bi-Encoder
class SAGEBiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, method='ip'):
        super(SAGEBiEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method
        
        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'ip':
            # 내적 계산 (inner product)
            ip_sim = torch.sum(x[a_idx] * x[b_idx], dim=1)
            # 정규화된 값으로 변환
            ip_prob = torch.sigmoid(ip_sim)  # 내적 값을 [0, 1] 범위로 매핑
            logits = ip_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'ip':
                # 내적 기반 손실 함수
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(ip_sim, labels.float())
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# GATv2 architecture for Bi-Encoder
class GATv2BiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, method='ip'):
        super(GATv2BiEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method
        
        
        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'ip':
            # 내적 계산 (inner product)
            ip_sim = torch.sum(x[a_idx] * x[b_idx], dim=1)
            # 정규화된 값으로 변환
            ip_prob = torch.sigmoid(ip_sim)  # 내적 값을 [0, 1] 범위로 매핑
            logits = ip_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'ip':
                # 내적 기반 손실 함수
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(ip_sim, labels.float())
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


from src.utils.encoder.crossencoder_utils import CrossEncoderModel

class NoGNNCrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(NoGNNCrossEncoderModel, self).__init__(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 1

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.vector_tensor

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # 내적 계산 (inner product)
        ip_sim = torch.sum(x[article1_idx] * x[article2_idx], dim=1)
        ip_sim = ip_sim.unsqueeze(1)  # 1차원 텐서를 2차원으로 변환

        combined_output = torch.cat([
            pooled_output,
            ip_sim
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)




class GCNCrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(GCNCrossEncoderModel, self).__init__(model_name)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)


        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 1

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # 내적 계산 (inner product)
        ip_sim = torch.sum(x[article1_idx] * x[article2_idx], dim=1)
        ip_sim = ip_sim.unsqueeze(1)  # 1차원 텐서를 2차원으로 변환

        combined_output = torch.cat([
            pooled_output,
            ip_sim
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)





class SAGECrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(SAGECrossEncoderModel, self).__init__(model_name)
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 1

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # 내적 계산 (inner product)
        ip_sim = torch.sum(x[article1_idx] * x[article2_idx], dim=1)
        ip_sim = ip_sim.unsqueeze(1)  # 1차원 텐서를 2차원으로 변환

        combined_output = torch.cat([
            pooled_output,
            ip_sim
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)



class GATv2CrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor,heads=8, method='cosine'):
        super(GATv2CrossEncoderModel, self).__init__(model_name)
        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)

        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 1

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # 내적 계산 (inner product)
        ip_sim = torch.sum(x[article1_idx] * x[article2_idx], dim=1)
        ip_sim = ip_sim.unsqueeze(1)  # 1차원 텐서를 2차원으로 변환

        combined_output = torch.cat([
            pooled_output,
            ip_sim
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
