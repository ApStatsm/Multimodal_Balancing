import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import glob
import os
import random
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1. 설정 (Config)
# =============================================================================
class Config:
    PROJECT_NAME = "lstm_multiclass_5emotions"
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    # 경로 설정
    DATA_DIR = '/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data'
    RESULT_DIR = './result'  # 결과 저장 경로
    
    LSTM_INPUT_DIM = 4   
    LSTM_HIDDEN_DIM = 64
    LSTM_MAX_LEN = 128   
    
    NUM_CLASSES = 5      
    
    EPOCHS = 10          
    BATCH_SIZE = 32      
    LEARNING_RATE = 3e-4
    
    SEED = 42

# =============================================================================
# 2. 유틸리티
# =============================================================================
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def pad_sequences_custom(sequences, maxlen, padding='pre', value=0.0):
    num_samples = len(sequences)
    sample_shape = np.asarray(sequences[0]).shape
    feature_dim = sample_shape[1] if len(sample_shape) > 1 else 1
    padded_array = np.full((num_samples, maxlen, feature_dim), value, dtype=np.float32)
    for idx, seq in enumerate(sequences):
        seq = np.asarray(seq, dtype=np.float32)
        length = min(len(seq), maxlen)
        if padding == 'post':
            padded_array[idx, :length] = seq[:length]
        else:
            padded_array[idx, -length:] = seq[-length:]
    return padded_array

# =============================================================================
# 3. 데이터셋
# =============================================================================
class SignalDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        self.signal_cols = ['EDA', 'TEMP', 'Valence', 'Arousal']
        
        self.class_map = {
            'neutral': 0,
            'surprise': 1,
            'angry': 2,
            'sad': 3,
            'happy': 4
        }
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        self.data = []

        search_pattern = os.path.join(data_dir, "Sess*.csv")
        file_paths = glob.glob(search_pattern)
        if not file_paths:
            search_pattern = os.path.join(data_dir, "sess*.csv")
            file_paths = glob.glob(search_pattern)
        
        print(f"[Dataset] {len(file_paths)}개 파일 로드 중...")
        
        all_raw_signals = []
        
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                grouped = df.groupby('Segment_ID')
                
                for _, group in grouped:
                    label_str = str(group['Emotion'].iloc[0]).lower()
                    
                    if label_str not in self.class_map:
                        continue
                        
                    label = self.class_map[label_str]
                    
                    segment_df = group[self.signal_cols].copy()
                    segment_df = segment_df.rolling(window=5, min_periods=1).mean()
                    sig_values = segment_df.values
                    
                    all_raw_signals.append(sig_values)
                    
                    self.data.append({
                        'raw_signal': sig_values,
                        'label': label
                    })
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if len(all_raw_signals) == 0:
            print("❌ 로드된 데이터가 없습니다.")
            return

        print("[Dataset] Scaling 계산 중...")
        all_values = np.concatenate(all_raw_signals, axis=0)
        global_mean = np.mean(all_values, axis=0)
        global_std = np.std(all_values, axis=0)
        global_std[global_std == 0] = 1.0
        
        self.processed_data = []
        temp_signals = []
        
        for item in self.data:
            norm_seq = (item['raw_signal'] - global_mean) / global_std
            temp_signals.append(norm_seq)
            
        padded_signals = pad_sequences_custom(temp_signals, self.config.LSTM_MAX_LEN, padding='pre')
        
        for i, item in enumerate(self.data):
            self.processed_data.append({
                'signal': padded_signals[i],
                'label': item['label']
            })
            
        labels = [d['label'] for d in self.processed_data]
        print(f"[Dataset] 완료. 총 데이터: {len(self.processed_data)}")
        for name, idx in self.class_map.items():
            count = labels.count(idx)
            print(f"   - {name} ({idx}): {count}개")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return {
            'signal_input': torch.tensor(self.processed_data[idx]['signal'], dtype=torch.float),
            'label': torch.tensor(self.processed_data[idx]['label'], dtype=torch.long) 
        }

# =============================================================================
# 4. 모델 (Attention 메커니즘 적용)
# =============================================================================
class LSTM_With_Attention(nn.Module):
    def __init__(self, config):
        super(LSTM_With_Attention, self).__init__()
        
        # 1. Bi-LSTM (기존과 동일)
        self.lstm = nn.LSTM(
            input_size=config.LSTM_INPUT_DIM, 
            hidden_size=config.LSTM_HIDDEN_DIM, 
            batch_first=True,
            num_layers=2, 
            bidirectional=True
        )
        
        # 2. Attention Layer (새로 추가됨)
        # LSTM의 '모든 시점' 출력을 보고 각 시점의 중요도 점수(Score)를 계산
        self.attention = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_DIM * 2, 64), # Hidden*2 (양방향) -> 64
            nn.Tanh(),                                 # 비선형성
            nn.Linear(64, 1)                           # 64 -> 1 (스칼라 점수)
        )
        
        # 3. Classifier
        # Attention으로 요약된 벡터(Context Vector)를 받아서 분류
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Linear(64, config.NUM_CLASSES) 
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        
        # (1) LSTM 통과
        # output: [Batch, Seq_Len, Hidden_Dim * 2] -> 모든 시점의 은닉 상태를 다 사용
        # (기존에는 hn인 마지막 hidden state만 썼음)
        output, (hn, _) = self.lstm(x)
        
        # (2) Attention Score 계산
        # attn_scores: [Batch, Seq_Len, 1]
        attn_scores = self.attention(output)
        
        # (3) Softmax로 확률 변환 (가중치 계산)
        # attn_weights: [Batch, Seq_Len, 1] (모두 더하면 1이 됨)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # (4) Context Vector 생성 (Weighted Sum)
        # 각 시점의 출력(output)에 가중치(attn_weights)를 곱해서 다 더함
        # context_vector: [Batch, Hidden_Dim * 2]
        context_vector = torch.sum(attn_weights * output, dim=1)
        
        # (5) 최종 분류
        logits = self.classifier(context_vector)
        
        return logits

# =============================================================================
# 5. 학습 및 평가 함수
# =============================================================================
def train_step(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        signals = batch['signal_input'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            signals = batch['signal_input'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, all_labels, all_preds

# =============================================================================
# 6. Main
# =============================================================================
def main():
    set_seed(Config.SEED)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    print(f"🚀 5-Class Emotion Classification | Device: {Config.DEVICE}")
    print(f"📂 결과 저장 경로: {Config.RESULT_DIR}")
    
    try:
        full_dataset = SignalDataset(Config.DATA_DIR, Config)
    except FileNotFoundError as e:
        print(e)
        return
    
    if len(full_dataset) == 0:
        return

    # Train/Test Split (8:2)
    dataset_indices = np.arange(len(full_dataset))
    dataset_labels = np.array([item['label'] for item in full_dataset.processed_data])
    
    train_idx, test_idx, _, _ = train_test_split(
        dataset_indices, dataset_labels, test_size=0.2, random_state=Config.SEED, stratify=dataset_labels
    )
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = LSTM_With_Attention(Config).to(torch.device(Config.DEVICE))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print("\n🔥 학습 시작...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_step(model, train_loader, criterion, optimizer, torch.device(Config.DEVICE))
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}")

    # =========================================================================
    # 🏁 [중요] 최종 결과 4가지 출력 및 저장
    # =========================================================================
    print("\n" + "="*50)
    print("📊 최종 평가 리포트 (Test Set)")
    print("="*50)

    # 1. 평가 실행
    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, torch.device(Config.DEVICE))
    
    # 2. 결과 텍스트 생성
    target_names = [full_dataset.idx_to_class[i] for i in range(Config.NUM_CLASSES)]
    unique_labels = sorted(list(set(y_true)))
    present_target_names = [target_names[i] for i in unique_labels]
    
    report = classification_report(y_true, y_pred, target_names=present_target_names, zero_division=0)
    
    # 3. 콘솔 출력
    print(f"1️⃣ Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"2️⃣ Final Test Loss     : {test_loss:.4f}")
    print("\n3️⃣ Classification Report:\n")
    print(report)
    
    # 4. Confusion Matrix 시각화 및 저장
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_target_names, yticklabels=present_target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Acc: {test_acc*100:.1f}%)')
    
    # 이미지 저장
    cm_path = os.path.join(Config.RESULT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\n4️⃣ Confusion Matrix 저장 완료: {cm_path}")
    plt.show() # 화면에도 띄우기

    # 5. 텍스트 결과 파일로 저장
    txt_path = os.path.join(Config.RESULT_DIR, 'test_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*30 + "\n")
        f.write(" Final Test Results \n")
        f.write("="*30 + "\n\n")
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"✅ 텍스트 리포트 저장 완료: {txt_path}")

if __name__ == "__main__":
    main()