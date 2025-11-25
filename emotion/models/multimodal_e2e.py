import torch.nn as nn
from models.kobert_encoder import KoBERTEncoder
from models.bio_encoder import BioLSTMEncoder
from models.fusion_model import CrossAttentionFusion

class MultimodalEndToEnd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = KoBERTEncoder()      # frozen
        self.bio_encoder = BioLSTMEncoder(
            input_dim=config["model"]["bio_input_dim"],
            hidden_dim=config["model"]["bio_hidden_dim"],
            output_dim=config["model"]["bio_output_dim"]
        )
        self.fusion = CrossAttentionFusion(
            bio_dim=config["model"]["bio_output_dim"],
            text_dim=768,
            hidden_dim=config["model"]["fusion_hidden_dim"],
            num_classes=config["model"]["num_classes"]  # 🔥 정답
        )

    def forward(self, text_input, bio_input):
        text_feat = self.text_encoder(
            text_input["input_ids"],
            text_input["attention_mask"],
            text_input["token_type_ids"]
        )
        bio_feat = self.bio_encoder(bio_input)
        return self.fusion(bio_feat, text_feat)
