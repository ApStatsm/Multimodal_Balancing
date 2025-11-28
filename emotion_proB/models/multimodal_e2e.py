# multimodal_e2e.py

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
            num_classes=config["model"]["num_classes"]
        )

        # 🔥 [추가] MAAN 기반: 단일 모달 보조 분류기 (L_text, L_bio 계산용)
        num_classes = config["model"]["num_classes"]
        # KoBERT 출력 차원: 768
        self.aux_text_classifier = nn.Linear(768, num_classes) 
        # BioLSTM 출력 차원: 64
        self.aux_bio_classifier = nn.Linear(config["model"]["bio_output_dim"], num_classes) 

    # 🔥 [수정] 4가지 출력 (final_logits, fused_feature, aux_text_logits, aux_bio_logits)을 반환
    def forward(self, text_input, bio_input):
        # 1. 인코더 실행
        text_feat = self.text_encoder(
            text_input["input_ids"],
            text_input["attention_mask"],
            text_input["token_type_ids"]
        )
        bio_feat = self.bio_encoder(bio_input)

        # 2. 보조 로짓 계산 (L_text, L_bio)
        aux_text_logits = self.aux_text_classifier(text_feat)
        aux_bio_logits = self.aux_bio_classifier(bio_feat)
        
        # 3. 융합 모델 실행 (final_logits와 L_cons 계산용 fused_feature 반환)
        final_logits, fused_feature = self.fusion(bio_feat, text_feat)

        # 4. 최종 결과 반환
        return final_logits, fused_feature, aux_text_logits, aux_bio_logits