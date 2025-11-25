import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder

# ======================================================================
# [수정됨] TensorFlow 없이 'pad_sequences' 기능을 직접 구현한 함수
# ======================================================================
def pad_sequences_custom(sequences, maxlen, padding='post', value=0.0):
    """
    Keras의 pad_sequences와 똑같은 역할을 하는 함수입니다.
    입력: 리스트 형태의 시퀀스들
    출력: 길이가 맞춰진 Numpy 배열
    """
    num_samples = len(sequences)
    # 1. 결과 담을 0으로 채워진 배열 생성
    # 시퀀스 내부의 feature 차원 확인 (여기서는 4)
    sample_shape = np.asarray(sequences[0]).shape
    feature_dim = sample_shape[1] if len(sample_shape) > 1 else 1
    
    # (샘플 수, 최대 길이, 특징 수)
    padded_array = np.full((num_samples, maxlen, feature_dim), value, dtype=np.float32)

    for idx, seq in enumerate(sequences):
        # 시퀀스가 maxlen보다 길면 자르고, 짧으면 그대로 둠
        seq = np.asarray(seq, dtype=np.float32)
        length = min(len(seq), maxlen)
        
        if padding == 'post':
            # 앞부분부터 채움 (뒤가 0이 됨)
            padded_array[idx, :length] = seq[:length]
        else:
            # 뒷부분부터 채움 (앞이 0이 됨)
            padded_array[idx, -length:] = seq[-length:]
            
    return padded_array

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        
        # 1. 데이터 로드 및 병합
        search_pattern = os.path.join(data_dir, "Sess*.csv")
        # 대소문자 이슈 방지를 위해 없는 경우 소문자로 한 번 더 시도
        file_paths = glob.glob(search_pattern)
        if not file_paths:
            search_pattern = os.path.join(data_dir, "sess*.csv")
            file_paths = glob.glob(search_pattern)
        
        print(f"[Dataset] '{data_dir}'에서 총 {len(file_paths)}개의 파일을 찾았습니다.")
        
        if not file_paths:
            raise FileNotFoundError(f"경로에 csv 파일이 없습니다. 경로를 확인해주세요: {data_dir}")

        all_dfs = []
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
            except Exception as e:
                print(f"파일 읽기 오류 ({path}): {e}")
        
        self.df = pd.concat(all_dfs, ignore_index=True)
        print(f"[Dataset] 데이터 통합 완료. 총 행 개수: {len(self.df)}")
        
        # -----------------------------------------------------------
        # [체크] 텍스트 컬럼명 확인 (Transcript / Text 등)
        self.text_col = 'Transcript'  
        self.signal_cols = ['EDA', 'TEMP', 'Valence', 'Arousal'] 
        # -----------------------------------------------------------

        # 2. Label Encoding
        self.le = LabelEncoder()
        self.df['Emotion'] = self.le.fit_transform(self.df['Emotion'].astype(str))
        
        # 3. 전처리 시작
        self.data = self._preprocess_data()
        print(f"[Dataset] 전처리 완료. 최종 데이터 개수: {len(self.data)}")

    def _preprocess_data(self):
        processed_data = []
        grouped = self.df.groupby('Segment_ID')
        
        raw_signals = []
        
        for segment_id, group in grouped:
            # 신호 추출
            signal_seq = group[self.signal_cols].values 
            raw_signals.append(signal_seq)
            
            # 텍스트 추출
            if self.text_col in group.columns:
                text_data = str(group[self.text_col].iloc[0])
            else:
                text_data = ""
            
            # 레이블 추출
            label = group['Emotion'].iloc[0]
            
            processed_data.append({
                'text': text_data,
                'label': label
            })
            
        # [수정됨] Keras 함수 대신 커스텀 함수 사용
        padded_signals = pad_sequences_custom(
            raw_signals, 
            maxlen=self.config.LSTM_MAX_LEN, 
            padding='post', 
            value=0.0
        )
        
        for i, item in enumerate(processed_data):
            item['signal'] = padded_signals[i]
            
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # KoBERT 토크나이징
        inputs = self.tokenizer(
            item['text'],
            return_tensors='pt',
            max_length=self.config.MAX_LEN,
            padding='max_length',
            truncation=True
        )
        
        signal_tensor = torch.tensor(item['signal'], dtype=torch.float)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'signal_input': signal_tensor,
            'label': label
        }

if __name__ == "__main__":
    from config import Config
    from transformers import BertTokenizer
    
    print("--- Dataset 테스트 ---")
    try:
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        dataset = MultimodalDataset(Config.DATA_DIR, tokenizer, Config)
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\n[성공] 샘플 데이터:")
            print(f" - 텍스트: {sample['input_ids'].shape}")
            print(f" - 신호: {sample['signal_input'].shape}")
            print(f" - 레이블: {sample['label']}")
    except Exception as e:
        print(f"[에러]: {e}")