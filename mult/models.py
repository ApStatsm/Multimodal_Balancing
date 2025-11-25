import torch
import torch.nn as nn
from transformers import BertModel

class LSTMFeatureExtractor(nn.Module):
    """
    [신호 데이터 처리 담당]
    기존 노트북의 구조: LSTM(64) -> Dropout -> Dense(32) -> ReLU
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(LSTMFeatureExtractor, self).__init__()
        
        # 1. LSTM 레이어
        # batch_first=True: 입력이 (Batch, Time, Feature) 순서임
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 2. 차원 축소 (64 -> 32)
        # 텍스트 벡터(768)가 너무 크니까, 신호 벡터도 적당히 압축해서 합침
        self.fc = nn.Linear(hidden_dim, 32) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> (32, 141, 4)
        
        # LSTM 통과
        # output: 모든 시점의 출력, (hn, cn): 마지막 은닉 상태
        _, (hn, _) = self.lstm(x) 
        
        # hn의 마지막 층만 가져옴: (Batch, hidden_dim) -> (32, 64)
        last_hidden = hn[-1] 
        
        # Dropout -> Dense -> ReLU
        x = self.dropout(last_hidden)
        x = self.fc(x) # (32, 32)
        x = self.relu(x)
        
        return x # 최종 32차원 벡터 반환

class MultimodalFusion(nn.Module):
    """
    [최종 합체 모델]
    KoBERT(텍스트) + LSTM(신호) => 분류기
    """
    def __init__(self, config):
        super(MultimodalFusion, self).__init__()
        
        # 1. 텍스트 모델 (KoBERT)
        self.text_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.text_drop = nn.Dropout(config.DROPOUT_RATE)
        
        # 2. 신호 모델 (위에서 만든 LSTM 클래스 사용)
        self.signal_encoder = LSTMFeatureExtractor(
            input_dim=config.LSTM_INPUT_DIM,   # 4
            hidden_dim=config.LSTM_HIDDEN_DIM, # 64
            dropout_rate=config.DROPOUT_RATE
        )
        
        # 3. 최종 분류기 (Fusion)
        # 입력 크기 = BERT(768) + LSTM(32) = 800
        fusion_input_dim = 768 + 32
        self.classifier = nn.Linear(fusion_input_dim, config.NUM_CLASSES) # 7개 감정

    def forward(self, input_ids, attention_mask, signal_input):
        # --- (1) 텍스트 처리 ---
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output: [CLS] 토큰의 벡터 (문장 전체 의미, 768차원)
        text_vec = self.text_drop(text_output.pooler_output) 
        
        # --- (2) 신호 처리 ---
        signal_vec = self.signal_encoder(signal_input) # 32차원
        
        # --- (3) 결합 (Concatenate) ---
        # dim=1 (가로 방향)으로 이어 붙임
        combined_vec = torch.cat((text_vec, signal_vec), dim=1) # 800차원
        
        # --- (4) 최종 예측 ---
        logits = self.classifier(combined_vec) # 7개 점수(Logits) 반환
        
        return logits