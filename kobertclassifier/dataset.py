# dataset.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
import chardet

# ===============================
# KoBERT Dataset Class
# ===============================
class KoBERTDataset(Dataset):
    printed_label_map = False

    def __init__(self, df, text_folder, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_folder = text_folder
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = []
        self.labels = []

        # 고정된 감정 매핑
        self.label_map = {
            "happy": 0,
            "surprise": 1,
            "angry": 2,
            "neutral": 3,
            "disgust": 4,
            "fear": 5,
            "sad": 6
        }

        # 라벨 매핑 로그 한 번만 출력
        if not KoBERTDataset.printed_label_map:
            print(" Using predefined label mapping:")
            for k, v in self.label_map.items():
                print(f"  {k:<15} → {v}")
            KoBERTDataset.printed_label_map = True

        #  Segment_ID 단위로 txt 1개만 로드
        for _, row in self.df.iterrows():
            seg_id = str(row["Segment_ID"]).strip()
            label_name = row["Emotion"]

            # 라벨 인덱스 변환
            label = self.label_map.get(label_name, 0)

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
                    if not text:
                        print(f" {seg_id}.txt 내용이 비어있습니다.")
                except Exception as e:
                    print(f" {txt_path} 읽기 실패 ({e})")
                    text = ""
            else:
                print(f" {seg_id}.txt 파일을 찾을 수 없습니다.")
                text = ""

            #  Segment_ID 단위로 1회만 추가
            self.texts.append(text)
            self.labels.append(label)

    # 필수 메서드 (DataLoader용)
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


# ===============================
#  Data Load Function
# ===============================
def load_data_from_folders(tokenizer, csv_path, text_folder,
                           test_size=0.1, val_size=0.1, batch_size=16, max_len=128):

    all_dfs = []

    #  CSV 경로가 폴더인지 확인
    if os.path.isdir(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        print(f" CSV folder detected: {csv_path}")
        print(f" Found {len(csv_files)} CSV files: {csv_files}")

        for fname in csv_files:
            full_path = os.path.join(csv_path, fname)
            with open(full_path, "rb") as f:
                raw = f.read()
                enc = chardet.detect(raw)["encoding"] or "cp949"

            try:
                df = pd.read_csv(full_path, encoding=enc)
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
                all_dfs.append(df)
                print(f" Loaded {fname} ({len(df)} rows)")
            except Exception as e:
                print(f"️ Failed to read {fname}: {e}")

    else:
        #  단일 CSV 파일 처리
        with open(csv_path, "rb") as f:
            raw = f.read()
            enc = chardet.detect(raw)["encoding"] or "cp949"
        df = pd.read_csv(csv_path, encoding=enc)
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
        all_dfs.append(df)
        print(f" Loaded single CSV: {csv_path} ({len(df)} rows)")

    #  여러 CSV 합치기
    if len(all_dfs) > 0:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f" Combined total rows: {len(df)}")
    else:
        raise ValueError(" No CSV files found in the specified path.")

    #  Segment_ID 기준 그룹화
    grouped = (
        df.groupby("Segment_ID")["Emotion"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    print(f"Aggregated by Segment_ID → {len(grouped)} unique segments")
    print(" Emotion distribution:\n", grouped["Emotion"].value_counts())

    #  그룹화된 DataFrame 기준으로 분할
    #  6:2:2 비율로 분할
    train_df, temp_df = train_test_split(grouped, test_size=0.4,stratify=grouped["Emotion"], random_state=42)  # 60% train, 40% 나머지
    val_df, test_df = train_test_split(temp_df, test_size=0.5,stratify=temp_df["Emotion"], random_state=42)  # 나머지 절반씩 20% / 20%

    print(f" Dataset split (6:2:2): Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    print(f" Dataset split (by Segment_ID): Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    #  데이터셋 개수 출력
    print(f" Dataset split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    #  DataFrame 전달
    train_set = KoBERTDataset(train_df, text_folder, tokenizer, max_len)
    val_set = KoBERTDataset(val_df, text_folder, tokenizer, max_len)
    test_set = KoBERTDataset(test_df, text_folder, tokenizer, max_len)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader