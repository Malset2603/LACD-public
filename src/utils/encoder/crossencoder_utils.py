

import polars as pl
import torch
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer


from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel

from src.utils.utils import article_key_function
from src.utils.encoder.utils import ThreeLayerClassifier

class RuleHierarchicalEncoderModel(torch.nn.Module):
    def __init__(self, model_name = "klue/roberta-base"):
        super(RuleHierarchicalEncoderModel, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)  # Rule 전용 Encoder 추가
        except:
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id

        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size

        transformer_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=2)

        self.classifier = torch.nn.Linear(hidden_size, 1)  # Article과 Rule의 인코딩을 결합하여 사용

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch_size, num_sentences, seq_length = input_ids.shape
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird", "roberta"] else outputs.last_hidden_state[:, -1, :]

        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)
        transformer_output = self.transformer_encoder(sentence_embeddings)  # Transformer 적용
        document_embedding = transformer_output.mean(dim=1)  # Mean pooling across sentences

        output = self.dropout(document_embedding)
        logits = self.classifier(output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            labels = labels.float().view_as(logits)  # 🔥 labels 크기를 logits과 동일하게 맞추기
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)  # 🔥 unsqueeze(-1) 제거하여 크기 맞춤

        return SequenceClassifierOutput(loss=loss, logits=logits)


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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
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

        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird", "roberta"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용


        rule_outputs = self.rule_encoder(input_ids=rule_input_ids, attention_mask=rule_attention_mask)
        rule_pooled_output = rule_outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird", "roberta"] else rule_outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용


        # TODO: pooled output 과 rule pooled output 을 합쳐서 combined output 을 생성
        combined_output = torch.cat([pooled_output, rule_pooled_output], dim=-1) 

        logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

class RuleCrossEncoderModelNew(torch.nn.Module):
    def __init__(self, model_name):
        super(RuleCrossEncoderModelNew, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.rule_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except:
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.rule_encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.rule_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
        self.rule_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
        self.rule_encoder.config.pad_token_id = self.rule_encoder.config.eos_token_id

        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size
        rule_hidden_size = self.rule_encoder.config.hidden_size

        # Article용 classifier와 Rule용 classifier를 각각 생성합니다.
        self.classifier_article = torch.nn.Linear(hidden_size, 1)
        self.classifier_rule = torch.nn.Linear(rule_hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        rule_input_ids=None,
        rule_attention_mask=None,
        labels=None,
    ):
        # Article 인코딩
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 표현

        # Rule 인코딩 (여기서는 rule_encoder를 사용합니다)
        rule_outputs = self.rule_encoder(input_ids=rule_input_ids, attention_mask=rule_attention_mask)
        rule_pooled_output = rule_outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 표현

        # 각각의 pooled output에 대해 별도의 classifier를 적용합니다.
        article_logits = self.classifier_article(pooled_output)  # [batch_size, 1]
        rule_logits = self.classifier_rule(rule_pooled_output)     # [batch_size, 1]

        # 두 logits를 평균하여 최종 output을 도출합니다.
        combined_logits = (article_logits + rule_logits) / 2

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(combined_logits, labels.float().unsqueeze(-1))

        return SequenceClassifierOutput(loss=loss, logits=combined_logits)

class CrossEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CrossEncoderModel, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except:
            # Model type should be one of ... error
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
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
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird", "roberta"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        
        pooled_output = self.dropout(pooled_output)


        logits = self.classifier(pooled_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))  # Ensure labels match logits shape

        return SequenceClassifierOutput(loss=loss, logits=logits)




class NLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, method="baseline"):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.method = method

        self.label_map = {True: 1, False: 0}


    def __len__(self):
        return self.dataframe.height
    
    def _get_row(self, index):
        return self.dataframe.row(index, named=True)

    def __getitem__(self, index):
        row = self._get_row(index)
        premise = row["article1"]
        hypothesis = row["article2"]
        label = 1 if row["answer"] else 0
        case_idx = row.get("case_idx", 0)
        rule_idx = row.get("rule_idx", 0)

        if self.method == "case-augmentation":
            from src.methods.case_augmentation.prompt import generate_case
            premise += "\ncase:\n" + generate_case(None, None, premise)
            hypothesis += "\ncase:\n" + generate_case(None, None, hypothesis, case_idx=case_idx)
        
        elif self.method == "case-concat-augmentation":
            from src.methods.case_augmentation.prompt import generate_case
            article_key = article_key_function(premise) + "-" + article_key_function(hypothesis)
            hypothesis = "case:\n" + generate_case(None, None, premise, article_key=article_key) + "\n" + hypothesis

        elif self.method == "rule-augmentation":
            from src.methods.rule_augmentation.prompt import generate_rule
            premise += "\nrule:\n" + generate_rule(None, None, premise, rule_idx=rule_idx)
            hypothesis += "\nrule:\n" + generate_rule(None, None, hypothesis, rule_idx=rule_idx)

        elif self.method == "rule-only":
            from src.methods.rule_augmentation.prompt import generate_rule
            premise = generate_rule(None, None, premise, rule_idx=rule_idx)
            hypothesis = generate_rule(None, None, hypothesis, rule_idx=rule_idx)

        # rule 과 article 을 별도로 encoding.
        elif self.method == "rule-augmentation-divide" or self.method == "rule-augmentation-divide-new":
            from src.methods.rule_augmentation.prompt import generate_rule

            rule_premise = generate_rule(None, None, premise, rule_idx=rule_idx)
            rule_hypothesis = generate_rule(None, None, hypothesis, rule_idx=rule_idx)
            # premise와 hypothesis는 그대로 사용, rule을 별도로 인코딩

            
            encoding = self.tokenizer.encode_plus(
                text=premise,
                text_pair=hypothesis,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            rule_encoding = self.tokenizer.encode_plus(
                text=rule_premise,
                text_pair=rule_hypothesis,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'rule_input_ids': rule_encoding['input_ids'].flatten(),
                'rule_attention_mask': rule_encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

        elif self.method == "rule-augmentation-hierarchical":
            from src.methods.rule_augmentation.prompt import generate_rule
            from torch.nn.utils.rnn import pad_sequence

            premise_rules = generate_rule(None, None, premise, rule_idx=rule_idx).split("IF:")[1:]
            hypothesis_rules = generate_rule(None, None, hypothesis, rule_idx=rule_idx).split("IF:")[1:]

            rules = []

            for pr in premise_rules:
                rule_encoding = self.tokenizer.encode_plus(
                    text=pr,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                rules.append(rule_encoding)
            
            rule_encoding = self.tokenizer.encode_plus(
                text="[SEP]",
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            rules.append(rule_encoding)
            for hr in hypothesis_rules:
                rule_encoding = self.tokenizer.encode_plus(
                    text=hr,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                rules.append(rule_encoding)

            if len(rules) > 10:
                rules = rules[:10]

            while len(rules) < 10:
                rule_encoding = self.tokenizer.encode_plus(
                    text="[PAD]",
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                rules.append(rule_encoding)
            
            assert(len(rules) == 10)

            # rule_input_ids_list와 rule_attention_mask_list를 일정한 길이로 맞추기
            rule_input_ids_list = [r["input_ids"].squeeze(0) for r in rules]
            rule_attention_mask_list = [r["attention_mask"].squeeze(0) for r in rules]


            # Tensor로 변환 (모든 샘플의 크기가 다를 경우 패딩 필요)
            rule_input_ids_padded = pad_sequence(rule_input_ids_list, batch_first=True, padding_value=0)
            rule_attention_mask_padded = pad_sequence(rule_attention_mask_list, batch_first=True, padding_value=0)

            return {
                "input_ids": rule_input_ids_padded,  # (num_rules, seq_len)
                "attention_mask": rule_attention_mask_padded,  # (num_rules, seq_len)
                "labels": torch.tensor(label, dtype=torch.float).unsqueeze(-1),  # (1,)
                "index": index  # 같은 원문에 대한 그룹핑을 위해 index 반환
            }


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
        true_count = self.dataframe.filter(pl.col("answer") == True).height
        false_count = self.dataframe.filter(pl.col("answer") == False).height
        print(f"True labels: {true_count}")
        print(f"False labels: {false_count}")
