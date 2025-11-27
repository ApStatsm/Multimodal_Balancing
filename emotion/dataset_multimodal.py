import os
import torch
import pandas as pd
import chardet
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 🔥 [수정] 사용자 지정 순서 적용
# 0: Neutral, 1: Surprise, 2: Angry, 3: Sad, 4: Happy
def get_label(emotion_str):
    e = emotion_str.lower()
    if "neutral" in e: return 0
    if "sur" in e: return 1      # surprise
    if "ang" in e: return 2      # angry, anger
    if "sad" in e: return 3      # sad, sadness
    if "hap" in e: return 4      # happy, happiness
    return -1 # 예외 처리

class MultimodalDataset(Dataset):
    def __init__(self, df, text_folder, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_folder = text_folder
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = []
        self.bio_features = []
        self.labels = []

        for _, row in self.df.iterrows():
            seg_id = str(row["Segment_ID"]).strip()
            
            # 라벨 매핑
            raw_emotion = str(row["Emotion"])
            label = get_label(raw_emotion)
            
            # 매핑되지 않는 데이터(fear, disgust 등)는 건너뜀
            if label == -1:
                continue

            bio_vals = [
                float(row["EDA"]),
                float(row["TEMP"]),
                float(row["Valence"]),
                float(row["Arousal"])
            ]

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
                    text = raw.decode(enc, errors="ignore").strip()
                except:
                    pass

            self.texts.append(text)
            self.bio_features.append(bio_vals)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        text_input = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
        }
        bio_input = torch.tensor(self.bio_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return text_input, bio_input, label

def load_data_frames(session_folder):
    csv_files = [f for f in os.listdir(session_folder) if f.endswith(".csv")]
    dfs = []
    for fname in csv_files:
        path = os.path.join(session_folder, fname)
        df = pd.read_csv(path)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    grouped = (
        df.groupby("Segment_ID")
        .agg({
            "EDA": "mean", "TEMP": "mean",
            "Valence": "mean", "Arousal": "mean",
            "Emotion": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        })
        .reset_index()
    )

    # 1. Fear, Disgust 제거
    exclude_emotions = ['fear', 'disgust']
    grouped = grouped[~grouped['Emotion'].str.lower().isin(exclude_emotions)]
    
    # 2. Target 컬럼 생성 (Stratify용)
    grouped["target"] = grouped["Emotion"].apply(lambda x: get_label(str(x)))
    
    # -1(매핑 실패) 제거
    grouped = grouped[grouped["target"] != -1]

    # Train(8) : Test(2) 분할
    train_df, test_df = train_test_split(
        grouped, test_size=0.2, stratify=grouped["target"], random_state=42
    )
    
    print(f"Data Filtered (5 Classes). Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df