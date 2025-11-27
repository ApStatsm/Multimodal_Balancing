import os
import torch
import pandas as pd
import chardet
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np # 🔥 numpy 추가

class MultimodalDataset(Dataset):
    # 🔥 weights 파라미터 추가 및 저장
    def __init__(self, df, text_folder, tokenizer, max_len=128, weights=None):
        self.df = df.reset_index(drop=True)
        self.text_folder = text_folder
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 🔥 가중치 설정: 제공된 가중치가 없으면 모두 1로 설정
        if weights is None:
            self.weights = np.ones(len(self.df), dtype=np.float32)
        else:
            if len(weights) != len(self.df):
                raise ValueError("Weights length must match DataFrame length.")
            self.weights = np.array(weights, dtype=np.float32)

        self.texts = []
        self.bio_features = []
        self.labels = []
        
        # ... (이하 기존 __init__ 로직 유지) ...
        for _, row in self.df.iterrows():
            seg_id = str(row["Segment_ID"]).strip()
            
            # 이진 분류: neutral(0) vs others(1)
            raw_emotion = row["Emotion"].lower()
            # config.py 기준으로 2클래스 이므로 0 또는 1
            label = 0 if raw_emotion == "neutral" else 1

            bio_vals = [
                float(row["EDA"]),
                float(row["TEMP"]),
                float(row["Valence"]),
                float(row["Arousal"])
            ]
            
            # 텍스트 파일 찾기 (기존 로직)
            txt_path = None
            for root, _, files in os.walk(self.text_folder):
                if f"{seg_id}.txt" in files:
                    txt_path = os.path.join(root, f"{seg_id}.txt")
                    break

            text = ""
            if txt_path:
                try:
                    with open(txt_path, "rb") as f:
                        raw = f.read()
                        enc = chardet.detect(raw)["encoding"] or "utf-8"
                    with open(txt_path, "r", encoding=enc) as f:
                        text = f.read().strip()
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
                    text = "[NO_TEXT]"
            else:
                 text = "[NO_TEXT]"

            self.texts.append(text)
            self.bio_features.append(bio_vals)
            self.labels.append(label)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        bio_input = torch.tensor(self.bio_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # 🔥 가중치 반환
        weight = torch.tensor(self.weights[idx], dtype=torch.float)
        
        text_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        
        # 텐서 차원 제거 (Batch=1)
        text_input = {k: v.squeeze(0) for k, v in text_input.items()}
        
        # 🔥 weight 반환값에 추가
        return text_input, bio_input, label, weight 

# load_data_frames 함수는 기존과 동일하게 유지
def load_data_frames(session_folder):
    """
    CSV를 읽고 전처리한 뒤, Train/Test DataFrame을 반환합니다.
    (Fear, Disgust 제외 로직 및 8:2 분할 로직)
    """
    # (생략: 기존 load_data_frames 함수 내용)
    csv_files = [f for f in os.listdir(session_folder) if f.endswith(".csv")]
    dfs = []
    for fname in csv_files:
        path = os.path.join(session_folder, fname)
        df = pd.read_csv(path)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 세션별 집계
    grouped = (
        df.groupby("Segment_ID")
        .agg({
            "EDA": "mean", "TEMP": "mean",
            "Valence": "mean", "Arousal": "mean",
            "Emotion": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        })
        .reset_index()
    )

    # Fear, Disgust 제거
    exclude_emotions = ['fear', 'disgust']
    grouped = grouped[~grouped['Emotion'].str.lower().isin(exclude_emotions)]
    
    print(f"Dataset Filtered: Removed Fear/Disgust. Total Samples: {len(grouped)}")

    # [수정 후] 🔥 8:2 분할 로직 (Train: 80%, Test: 20%)
    train_df, test_df = train_test_split(
        grouped, 
        test_size=0.2, 
        random_state=42, 
        stratify=grouped['Emotion']
    )

    print(f"Data Split: Train={len(train_df)}, Test={len(test_df)}")
    
    # 🔥 반환값에서 val_df 제거
    return train_df, test_df