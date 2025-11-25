# multimodal_project/
# │
# ├── data/                 # 데이터 파일 (.csv, .txt 등) 저장
# ├── saved_models/         # 학습된 모델(.pt) 저장
# │
# ├── config.py             # [중요] 하이퍼파라미터, 경로 등 설정값 모음
# ├── dataset.py            # 데이터 로드 및 전처리 (Dataset 클래스)
# ├── models.py             # 멀티모달 모델 구조 정의 (Model 클래스)
# ├── train.py              # 학습(Train) 및 검증(Valid) 로직
# ├── utils.py              # 시드 고정, 평가지표 계산 등 보조 함수
# └── main.py               # 프로그램 실행 진입점

import torch
import numpy as np
import random
import os

def set_seed(seed_value=42):
    """
    실험 결과 재현을 위해 시드(Seed)를 고정하는 함수
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    # 맥(MPS) 환경에서도 시드 고정이 어느 정도 적용됩니다.
    # CuDNN 결정론적 옵션 (GPU 관련)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Utils] Random Seed가 {seed_value}로 고정되었습니다.")

def save_checkpoint(model, path):
    """모델 가중치 저장"""
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    """모델 가중치 로드"""
    model.load_state_dict(torch.load(path))
