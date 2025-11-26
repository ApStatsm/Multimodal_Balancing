import torch.nn as nn

class BioLSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 🔥 [수정 5] Layer Norm 추가
        self.ln = nn.LayerNorm(output_dim)

        # 🔥 [수정 2] Unfreeze: 파라미터 동결 코드 제거됨 (학습 가능)

    def forward(self, x):
        # x: (B, 4) -> (B, 1, 4)
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]         
        out = self.fc(h_last)
        return self.ln(out)    # Apply Layer Norm