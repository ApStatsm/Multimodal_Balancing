import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import chardet

class MultimodalDataset(Dataset):
    def __init__(self, df, text_folder, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_folder = text_folder
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = []
        self.bio_features = []
        self.labels = []

        # ---------------------------------------------------------
        # [수정 1] 이진 분류 라벨링 (Binary Labeling)
        # Neutral = 0
        # Biased (Happy, Surprise, Angry, Disgust, Fear, Sad) = 1
        # ---------------------------------------------------------
        for _, row in self.df.iterrows():
            seg_id = str(row["Segment_ID"]).strip()
            emotion = row["Emotion"].lower() # 소문자 처리

            # 0: Neutral, 1: Biased
            if emotion == 'neutral':
                label = 0
            else:
                label = 1 

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
                    print(f"[ERROR] Cannot read {txt_path}")

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


def load_multimodal_data(tokenizer, session_folder, text_folder,
                         batch_size=16, max_len=128):
    """
    모든 데이터를 로드하여 하나의 DataLoader로 반환합니다.
    (Split은 main.py의 K-Fold 로직에서 수행)
    """

    csv_files = [f for f in os.listdir(session_folder) if f.endswith(".csv")]
    dfs = []

    for fname in csv_files:
        path = os.path.join(session_folder, fname)
        df = pd.read_csv(path)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV files found in {session_folder}")

    df = pd.concat(dfs, ignore_index=True)

    # 데이터 집계 (Segment_ID 기준)
    grouped = (
        df.groupby("Segment_ID")
        .agg({
            "EDA": "mean",
            "TEMP": "mean",
            "Valence": "mean",
            "Arousal": "mean",
            "Emotion": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        })
        .reset_index()
    )

    # ------- [수정 2] 분포 확인 (Binary 기준) -------
    # 임시로 라벨링 변환해서 분포 출력
    temp_labels = grouped["Emotion"].apply(lambda x: 0 if x.lower() == 'neutral' else 1)
    print("\n📌 FULL DATASET Distribution (Binary):")
    print(f"   Total Samples: {len(grouped)}")
    print(f"   0 (Neutral)  : {sum(temp_labels == 0)}")
    print(f"   1 (Biased)   : {sum(temp_labels == 1)}")

    # ------- [수정 3] Split 없이 전체 데이터셋 반환 -------
    full_set = MultimodalDataset(grouped, text_folder, tokenizer, max_len)
    
    # 3개를 반환하던 기존 main.py와의 호환성을 위해 
    # (train, val, test) 형식으로 반환하지만, 여기서는 전체 데이터 1개만 반환합니다.
    full_loader = DataLoader(full_set, batch_size=batch_size, shuffle=False)

    return full_loader, None, None