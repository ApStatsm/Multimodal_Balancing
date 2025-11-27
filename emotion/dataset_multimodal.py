import os
import torch
import pandas as pd
import chardet
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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
            
            # 이진 분류: neutral(0) vs others(1)
            raw_emotion = row["Emotion"].lower()
            label = 0 if raw_emotion == "neutral" else 1

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
    """
    CSV를 읽고 전처리한 뒤, Train(80%)/Test(20%) DataFrame을 반환합니다.
    (Fear, Disgust 제외 로직 추가됨)
    """
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

    # 🔥 [추가] Fear, Disgust 제거
    # 제외할 감정 목록 정의
    exclude_emotions = ['fear', 'disgust']
    
    # 해당 감정이 포함되지 않은 데이터만 남김 (~ 연산자 사용)
    # 대소문자 문제 방지를 위해 .str.lower() 사용
    grouped = grouped[~grouped['Emotion'].str.lower().isin(exclude_emotions)]
    
    print(f"Dataset Filtered: Removed {exclude_emotions}. Current Size: {len(grouped)}")

    # 이진 분류를 위한 Stratify 기준 생성
    grouped["target"] = grouped["Emotion"].apply(lambda x: 0 if x.lower()=="neutral" else 1)

    # Train(8) : Test(2) 분할
    train_df, test_df = train_test_split(
        grouped, test_size=0.2, stratify=grouped["target"], random_state=42
    )
    
    print(f"Data Loaded: Train {len(train_df)}, Test {len(test_df)}")
    return train_df, test_df