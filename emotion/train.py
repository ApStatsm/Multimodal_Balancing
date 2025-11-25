import torch
import torch.nn as nn
import time
from tqdm import tqdm

def run_epoch(model, loader, optimizer, device, mode="train"):
    """
    1 Epoch 동안 학습(Train) 또는 검증(Val)을 수행하는 함수
    
    Returns:
        acc (float): 정확도
        avg_loss (float): 평균 손실값
        elapsed_time (float): 걸린 시간
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    
    # Loss 함수 정의 (이진 분류지만 CrossEntropy 사용 가능)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    # Gradient 계산 여부 설정
    with torch.set_grad_enabled(mode == "train"):
        # Tqdm으로 진행률 표시 (leave=False: 완료 후 줄 삭제로 깔끔하게)
        pbar = tqdm(loader, desc=mode.upper(), leave=False)
        
        for text_input, bio_input, label in pbar:
            
            # 1. 데이터 장치 이동
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # 2. 초기화
            if mode == "train":
                optimizer.zero_grad()

            # 3. 모델 예측 (Forward)
            logits = model(text_input, bio_input)
            
            # 4. Loss 계산
            loss = criterion(logits, label)

            # 5. 역전파 및 가중치 갱신 (Backward)
            if mode == "train":
                loss.backward()
                optimizer.step()

            # 6. 정확도 계산
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()
            
            # Tqdm 바에 현재 Loss 표시
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    acc = correct / total
    avg_loss = total_loss / len(loader)
    elapsed_time = time.time() - start_time

    return acc, avg_loss, elapsed_time


def evaluate_with_shuffle(model, loader, device, criterion, shuffle_bio=False):
    """
    테스트 단계에서 호출되는 함수.
    shuffle_bio=True일 경우, 생체신호(Bio)를 배치 내에서 섞어서 편향성을 테스트함.

    Returns:
        acc (float): 정확도
        avg_loss (float): 평균 손실값
        all_true (list): 실제 정답 리스트 (Confusion Matrix용)
        all_pred (list): 예측값 리스트 (Confusion Matrix용)
    """
    model.eval()
    
    total = 0
    correct = 0
    total_loss = 0.0
    
    all_true = []
    all_pred = []

    with torch.no_grad():
        for text_input, bio_input, label in loader:
            
            # 1. 데이터 장치 이동
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # -------------------------------------------------------
            # 🎲 [핵심] Bio Signal Shuffling Logic
            # -------------------------------------------------------
            if shuffle_bio:
                # 현재 배치 크기만큼 랜덤 인덱스 생성 (예: [3, 0, 2, 1])
                idx = torch.randperm(bio_input.size(0))
                # Bio 신호 순서를 섞어버림 (Text와 Label은 고정)
                # 이렇게 하면 Text만 정상이고 Bio는 노이즈가 됨
                bio_input = bio_input[idx]
            # -------------------------------------------------------

            # 2. 모델 예측
            logits = model(text_input, bio_input)
            
            # 3. Loss 계산
            loss = criterion(logits, label)
            total_loss += loss.item()

            # 4. 정확도 및 결과 수집
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            # 리스트에 결과 누적 (main.py에서 Confusion Matrix 그릴 때 사용)
            all_true.extend(label.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    acc = correct / total
    avg_loss = total_loss / len(loader)
    
    return acc, avg_loss, all_true, all_pred