import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import AutoModel


from src.utils.encoder.crossencoder_utils import CrossEncoderModel
from src.utils.encoder.utils import ThreeLayerClassifier

class NoGNNCrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(NoGNNCrossEncoderModel, self).__init__(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size +  in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.vector_tensor

        # Encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # Combine the CLS token embedding with the article embeddings
        combined_output = torch.cat([pooled_output, x[article_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
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

        self.query_encoder_model = AutoModel.from_pretrained("klue/roberta-base")
        for param in self.query_encoder_model.parameters():
            param.requires_grad = False

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article_idx, query_input_ids=None, query_attention_mask=None, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        query_outputs = self.query_encoder_model(input_ids=query_input_ids,attention_mask=query_attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        query_output =  query_outputs.last_hidden_state[:, 0, :]
        query_output = self.dropout(query_output)

        # Combine the CLS token embedding with the article embeddings
        combined_output = torch.cat([pooled_output, x[article_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)