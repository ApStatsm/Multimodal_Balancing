import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, bio_dim=32, text_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()

        self.bio_proj = nn.Linear(bio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 🔥 [수정 5] Layer Norm 추가
        self.ln_bio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 여기도 추가
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, bio, text):
        # Projection & Norm
        bio = self.ln_bio(self.bio_proj(bio)).unsqueeze(1)
        text = self.ln_text(self.text_proj(text)).unsqueeze(1)

        attn_bio, _ = self.attn(bio, text, text)
        attn_text, _ = self.attn(text, bio, bio)

        fused = torch.cat([attn_bio.squeeze(1), attn_text.squeeze(1)], dim=-1)
        return self.classifier(fused)