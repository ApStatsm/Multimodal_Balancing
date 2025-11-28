# fusion_model.py

import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, bio_dim=32, text_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()

        # 1. 차원 맞추기 (Projection)
        self.bio_proj = nn.Linear(bio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 학습 안정화를 위한 Layer Norm
        self.ln_bio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)

        # 2. Cross Attention
        # (Bio <-> Text 서로 정보를 교환)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # 3. 분류기 (Classifier)
        # 융합된 벡터(Concat)를 받아서 최종 예측
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Concat 했으므로 입력이 hidden_dim * 2
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    # 🔥 [수정] 융합 임베딩(fused_feature)을 추가로 반환
    def forward(self, bio, text):
        # [Step 1] Projection & Unsqueeze (Attention 입력을 위해 차원 추가)
        # (Batch, Dim) -> (Batch, 1, Hidden)
        bio = self.ln_bio(self.bio_proj(bio)).unsqueeze(1)
        text = self.ln_text(self.text_proj(text)).unsqueeze(1)

        # [Step 2] Cross Attention
        # Q(질문): Bio, K/V(정보): Text -> Bio 입장에서 Text 정보를 받음
        attn_output_bio, _ = self.attn(bio, text, text) 
        # Q(질문): Text, K/V(정보): Bio -> Text 입장에서 Bio 정보를 받음
        attn_output_text, _ = self.attn(text, bio, bio) 
        
        # [Step 3] 융합 벡터 생성 (Concat)
        # fused_feature: L_cons 계산에 사용되는 융합 임베딩
        fused_feature = torch.cat([
            attn_output_bio.squeeze(1), 
            attn_output_text.squeeze(1)
        ], dim=-1) # (B, Hidden*2)

        # [Step 4] 최종 분류
        final_logits = self.classifier(fused_feature) # (B, num_classes)

        return final_logits, fused_feature