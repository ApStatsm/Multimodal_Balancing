# dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
import chardet
import re

# ===============================
# KoBERT Dataset Class (이진 분류용 수정)
# ===============================
class KoBERTDataset(Dataset):
    printed_label_map = False

    def __init__(self, df, text_folder, tokenizer, max_len=128):
        # 1. 사용할 감정 정의 (fear, disgust 제외)
        target_emotions = ["neutral", "surprise", "angry", "sad", "happy"]
        
        # 2. 해당 감정만 있는 행만 필터링
        df_filtered = df[df["Emotion"].isin(target_emotions)].copy()
        self.df = df_filtered.reset_index(drop=True)
        
        self.text_folder = text_folder
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = []
        self.labels = []

        # 3. 이진 분류 매핑 (neutral=0, 나머지=1)
        self.label_map = {
            "neutral": 0,
            "surprise": 1,
            "angry": 1,
            "sad": 1,
            "happy": 1
        }

        # 라벨 매핑 로그 (한 번만 출력)
        if not KoBERTDataset.printed_label_map:
            print(" Using Binary label mapping (Neutral vs Biased):")
            print("  neutral  → 0")
            print("  surprise → 1 (biased)")
            print("  angry    → 1 (biased)")
            print("  sad      → 1 (biased)")
            print("  happy    → 1 (biased)")
            KoBERTDataset.printed_label_map = True

        # Segment_ID 단위로 데이터 로드
        for _, row in self.df.iterrows():
            seg_id = str(row["Segment_ID"]).strip()
            label_name = row["Emotion"]

            # 매핑된 라벨 가져오기 (0 또는 1)
            label = self.label_map[label_name]

            # txt 파일 경로 탐색
            txt_path = None
            for root, _, files in os.walk(text_folder):
                if f"{seg_id}.txt" in files:
                    txt_path = os.path.join(root, f"{seg_id}.txt")
                    break

            # 파일 로드
            if txt_path:
                try:
                    with open(txt_path, "rb") as f:
                        raw = f.read()
                        enc = chardet.detect(raw)["encoding"] or "cp949"
                    text = raw.decode(enc, errors="ignore").strip()
                except Exception:
                    text = ""
            else:
                text = ""

            # 텍스트 전처리
            clean_text = re.sub(r"[^가-힣a-zA-Z0-9\s\.\,\?\!]", "", text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            self.texts.append(clean_text)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text
        }

# load_data_from_folders 함수는 기존과 동일하게 유지해도 되지만, 
# 데이터셋 분할 시 'stratify'가 정상 작동하도록 주의해야 합니다. 
# (Dataset 클래스 내부에서 필터링하므로, 원본 DataFrame 단계에서는 그대로 둬도 되지만
#  명확하게 하기 위해 split 전에 필터링하는 것이 더 좋습니다.)

def load_data_from_folders(tokenizer, csv_path, text_folder,
                           test_size=0.1, val_size=0.1, batch_size=16, max_len=128):

    # ... (CSV 로드 부분은 기존과 동일) ...
    all_dfs = []
    if os.path.isdir(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        for fname in csv_files:
            full_path = os.path.join(csv_path, fname)
            with open(full_path, "rb") as f:
                raw = f.read()
                enc = chardet.detect(raw)["encoding"] or "cp949"
            try:
                df = pd.read_csv(full_path, encoding=enc)
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
                all_dfs.append(df)
            except Exception as e:
                print(f"️ Failed to read {fname}: {e}")
    else:
        with open(csv_path, "rb") as f:
            raw = f.read()
            enc = chardet.detect(raw)["encoding"] or "cp949"
        df = pd.read_csv(csv_path, encoding=enc)
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
        all_dfs.append(df)

    if len(all_dfs) > 0:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        raise ValueError(" No CSV files found.")

    # 1. Segment_ID 기준 그룹화 (최빈값 등)
    grouped = (
        df.groupby("Segment_ID")["Emotion"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    # ========================================================
    # [수정] Split 전에 미리 불필요한 감정 제거 (Stratify 오류 방지)
    # ========================================================
    target_emotions = ["neutral", "surprise", "angry", "sad", "happy"]
    grouped = grouped[grouped["Emotion"].isin(target_emotions)].copy()
    
    print(f"Filtered Data Count: {len(grouped)}")

    # 2. Train / Test 분할
    train_df, test_df = train_test_split(
        grouped,
        test_size=0.2,
        stratify=grouped["Emotion"],  # 원래 감정 비율대로 쪼갠 뒤 Dataset 내부에서 합침
        random_state=42
    )

    print(f" Dataset split (8:2): Train {len(train_df)}, Test {len(test_df)}")

    # DataFrame → Dataset (여기서 이진 라벨링 적용됨)
    train_set = KoBERTDataset(train_df, text_folder, tokenizer, max_len)
    test_set = KoBERTDataset(test_df, text_folder, tokenizer, max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader