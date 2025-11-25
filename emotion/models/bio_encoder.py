import torch.nn as nn

class BioLSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 🔥 BIO LSTM 동결 (바닐라 멀티모달)
        # for p in self.lstm.parameters():
        #     p.requires_grad = False
        # for p in self.fc.parameters():
        #     p.requires_grad = False

        # freeze LSTM? → 선택
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        """
        x: (B,4) → (B,1,4)
        """
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        h_last = h[-1]         # (B, hidden_dim)
        return self.fc(h_last) # (B,32)
