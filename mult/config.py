import torch

class Config:
    PROJECT_NAME = "kobert_lstm_multimodal"
    
    # =================================================================
    # [Mac 최적화] MPS(Metal) 가속 설정
    # M1/M2/M3 맥북이라면 'mps'가 잡혀야 GPU를 사용합니다.
    # =================================================================
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("✅ Apple Silicon GPU(MPS)를 사용합니다. 학습이 빨라집니다!")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        print("⚠️ GPU를 찾을 수 없어 CPU를 사용합니다. 속도가 느릴 수 있습니다.")
    
    # -----------------------------------------------------------------
    # [데이터 경로 설정]
    # 아까 보여주신 노트북의 경로를 그대로 넣었습니다.
    # 폴더가 실제로 존재하는지 꼭 확인해주세요!
    # -----------------------------------------------------------------
    DATA_DIR = '/Users/apstat/Desktop/02_연구/01_멀티모달 밸런싱/데이터'
    
    # 모델 저장 경로 (폴더가 없으면 에러나니 미리 만들어두거나 코드로 생성해야 함)
    MODEL_SAVE_PATH = "./saved_models/best_model.pt"
    
    # 텍스트(KoBERT) 설정
    BERT_MODEL_NAME = "monologg/kobert" 
    MAX_LEN = 64         
    
    # 신호(LSTM) 설정
    LSTM_INPUT_DIM = 4   # EDA, TEMP, Valence, Arousal
    LSTM_HIDDEN_DIM = 64
    LSTM_MAX_LEN = 141   
    
    # 학습 파라미터
    NUM_CLASSES = 7      
    EPOCHS = 2
    BATCH_SIZE = 32      # 맥에서는 메모리에 따라 16으로 줄여야 할 수도 있습니다.
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.3

#----------------------------------------------------------------------------------

# class Config:
#     PROJECT_NAME = "kobert_lstm_multimodal"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # [수정됨] 구체적인 파일명이 아니라, CSV 파일들이 들어있는 '폴더 경로'를 입력
#     DATA_DIR = '/Users/apstat/Desktop/연구/멀티모달 밸런싱/데이터' 
#     MODEL_SAVE_PATH = "./saved_models/best_model.pt"
    
#     # 텍스트(KoBERT) 관련
#     BERT_MODEL_NAME = "monologg/kobert" 
#     MAX_LEN = 64         # 텍스트 최대 토큰 길이
    
#     # 신호(LSTM) 관련
#     LSTM_INPUT_DIM = 4   # EDA, TEMP, Valence, Arousal
#     LSTM_HIDDEN_DIM = 64
#     LSTM_MAX_LEN = 141   # 노트북에서 확인한 max_len 값
    
#     # 학습 설정
#     NUM_CLASSES = 7      # 감정 개수
#     EPOCHS = 20
#     BATCH_SIZE = 32
#     LEARNING_RATE = 2e-5
#     DROPOUT_RATE = 0.3

