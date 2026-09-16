import torch
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel

from src.utils.utils import article_key_function
from src.utils.encoder.utils import ThreeLayerClassifier


class RuleCrossEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(RuleCrossEncoderModel, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)  # Rule 전용 Encoder 추가
            self.rule_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)  # Rule 전용 Encoder 추가
        except:
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.rule_encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
        self.rule_encoder.config.pad_token_id = self.rule_encoder.config.eos_token_id

        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = torch.nn.Linear(hidden_size * 2, 1)  # Article과 Rule의 인코딩을 결합하여 사용

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        rule_input_ids=None,
        rule_attention_mask=None,
        labels=None,
    ):

        # Rule 인코딩
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용


        rule_outputs = self.rule_encoder(input_ids=rule_input_ids, attention_mask=rule_attention_mask)
        rule_pooled_output = rule_outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용


        # TODO: pooled output 과 rule pooled output 을 합쳐서 combined output 을 생성
        combined_output = torch.cat([pooled_output, rule_pooled_output], dim=-1) 

        logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
    


class CrossEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CrossEncoderModel, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except:
            # Model type should be one of ... error
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id

        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size
        # Linear layer for classification
        self.classifier = torch.nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        # Input will now be concatenated pairs of sequences (cross-encoder)
        # Example: [CLS] seqA tokens ... [SEP] seqB tokens ...
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        
        pooled_output = self.dropout(pooled_output)


        logits = self.classifier(pooled_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))  # Ensure labels match logits shape

        return SequenceClassifierOutput(loss=loss, logits=logits)
