import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, bio_dim=32, text_dim=768, hidden_dim=256, num_classes=7):
        super().__init__()

        self.bio_proj = nn.Linear(bio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # ✨ 추가: 데이터 분포를 정규화해 학습 안정화
            nn.ReLU(),                   # ✨ 변경: ReLU보다 BERT와 궁합이 좋은 GELU
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, bio, text):
        bio = self.bio_proj(bio).unsqueeze(1)   # (B,1,256)
        text = self.text_proj(text).unsqueeze(1)

        attn_bio, _ = self.attn(bio, text, text)
        attn_text, _ = self.attn(text, bio, bio)

        fused = torch.cat([attn_bio.squeeze(1), attn_text.squeeze(1)], dim=-1)
        return self.classifier(fused)
