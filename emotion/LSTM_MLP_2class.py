import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import glob
import os
import random
# [추가] roc_auc_score, roc_curve 임포트
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1. 설정 (Config)
# =============================================================================
class Config:
    PROJECT_NAME = "lstm_binary_neutral_vs_biased"
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    DATA_DIR = '/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data'
    RESULT_DIR = './result'
    
    LSTM_INPUT_DIM = 4   
    LSTM_HIDDEN_DIM = 64
    LSTM_MAX_LEN = 128   
    
    NUM_CLASSES = 2  # 이진 분류
    
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
        
        self.target_labels = {
            0: 'Neutral',
            1: 'Biased'
        }
        
        self.emotion_mapping = {
            'neutral': 0,
            'surprise': 1,
            'angry': 1,
            'sad': 1,
            'happy': 1
        }
        
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
                    
                    if label_str not in self.emotion_mapping:
                        continue
                        
                    label = self.emotion_mapping[label_str]
                    
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
        
        for idx, name in self.target_labels.items():
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
# 4. 모델
# =============================================================================
class LSTM_Binary_Model(nn.Module):
    def __init__(self, config):
        super(LSTM_Binary_Model, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config.LSTM_INPUT_DIM, 
            hidden_size=config.LSTM_HIDDEN_DIM, 
            batch_first=True,
            num_layers=2, 
            bidirectional=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Linear(64, config.NUM_CLASSES) 
        )

    def forward(self, x):
        output, (hn, _) = self.lstm(x)
        last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        logits = self.mlp(last_hidden)
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
    all_probs = [] # [추가] 확률값 저장용
    
    with torch.no_grad():
        for batch in loader:
            signals = batch['signal_input'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # 확률 계산 (Softmax)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # 확률값 저장
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, all_labels, all_preds, all_probs

# =============================================================================
# 6. Main
# =============================================================================
def main():
    set_seed(Config.SEED)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    print(f"🚀 Binary Classification (Neutral vs Biased) + AUC | Device: {Config.DEVICE}")
    print(f"📂 결과 저장 경로: {Config.RESULT_DIR}")
    
    try:
        full_dataset = SignalDataset(Config.DATA_DIR, Config)
    except FileNotFoundError as e:
        print(e)
        return
    
    if len(full_dataset) == 0:
        return

    dataset_indices = np.arange(len(full_dataset))
    dataset_labels = np.array([item['label'] for item in full_dataset.processed_data])
    
    train_idx, test_idx, _, _ = train_test_split(
        dataset_indices, dataset_labels, test_size=0.2, random_state=Config.SEED, stratify=dataset_labels
    )
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = LSTM_Binary_Model(Config).to(torch.device(Config.DEVICE))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print("\n🔥 학습 시작...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_step(model, train_loader, criterion, optimizer, torch.device(Config.DEVICE))
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}")

    # =========================================================================
    # 🏁 최종 결과 출력 및 저장 (AUC 포함)
    # =========================================================================
    print("\n" + "="*50)
    print("📊 최종 평가 리포트 (Binary Classification)")
    print("="*50)

    # 1. 평가 실행 (확률값 y_probs 추가 반환)
    test_loss, test_acc, y_true, y_pred, y_probs = evaluate_model(model, test_loader, criterion, torch.device(Config.DEVICE))
    
    # 2. [추가] AUC 계산
    # y_probs는 [N, 2] 형태. Biased(Class 1)일 확률은 2번째 컬럼
    y_probs = np.array(y_probs)
    pos_probs = y_probs[:, 1] 
    
    try:
        auc_score = roc_auc_score(y_true, pos_probs)
    except ValueError:
        auc_score = 0.0
        print("⚠️ Warning: AUC calculation failed (Only one class present in y_true?)")

    # 3. 리포트 생성
    target_names = ['Neutral', 'Biased']
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    # 4. 콘솔 출력
    print(f"1️⃣ Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"2️⃣ Final Test AUC      : {auc_score:.4f}") # AUC 출력 추가
    print(f"3️⃣ Final Test Loss     : {test_loss:.4f}")
    print("\n4️⃣ Classification Report:\n")
    print(report)
    
    # 5. [추가] ROC Curve 시각화 및 저장
    if auc_score > 0:
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(Config.RESULT_DIR, 'roc_curve.png')
        plt.savefig(roc_path)
        print(f"\n5️⃣ ROC Curve 저장 완료: {roc_path}")
        plt.show()

    # 6. Confusion Matrix 시각화 및 저장
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Acc: {test_acc*100:.1f}%)')
    
    cm_path = os.path.join(Config.RESULT_DIR, 'confusion_matrix_binary.png')
    plt.savefig(cm_path)
    print(f"6️⃣ Confusion Matrix 저장 완료: {cm_path}")
    plt.show()

    # 7. 텍스트 결과 파일로 저장
    txt_path = os.path.join(Config.RESULT_DIR, 'test_results_binary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*30 + "\n")
        f.write(" Binary Classification Results \n")
        f.write(" (Neutral vs Biased) \n")
        f.write("="*30 + "\n\n")
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Test AUC      : {auc_score:.4f}\n") # AUC 저장 추가
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"✅ 텍스트 리포트 저장 완료: {txt_path}")

if __name__ == "__main__":
    main()