import torch.nn as nn
from transformers import BertModel

class KoBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")

        # 🔥 KoBERT 완전 동결
        for p in self.bert.parameters():
            p.requires_grad = False


    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        return out.pooler_output  # (B,768)
