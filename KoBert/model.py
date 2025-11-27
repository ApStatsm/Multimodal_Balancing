# model.py
import torch.nn as nn
from transformers import BertModel

class KoBERTClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        #self.dropout = nn.Dropout(dropout)
        #self.classifier = nn.Linear(768, num_classes)
        # 🔥 Dropout 제거한 MLP 분류기
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),  # 768 → 64
            nn.ReLU(),  # 활성화
            nn.Linear(64, num_classes)  # 64 → 5
        )
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs.pooler_output
        #x = self.dropout(cls)
        return self.classifier(cls)
