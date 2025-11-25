# model.py
import torch.nn as nn
from transformers import BertModel

class KoBERTClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs.pooler_output
        x = self.dropout(cls)
        return self.classifier(x)
