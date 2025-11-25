import torch
import torch.nn as nn
import time
from tqdm import tqdm

def run_epoch(model, loader, optimizer, device, mode="train"):
    """
    1 Epoch 동안 학습(Train) 또는 검증(Val)을 수행하는 함수
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    with torch.set_grad_enabled(mode == "train"):
        # dynamic_ncols=True로 창 크기 자동 조절
        pbar = tqdm(loader, desc=mode.upper(), leave=False, dynamic_ncols=True, mininterval=0.1)
        
        for text_input, bio_input, label in pbar:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            if mode == "train":
                optimizer.zero_grad()

            logits = model(text_input, bio_input)
            loss = criterion(logits, label)

            if mode == "train":
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    acc = correct / total
    avg_loss = total_loss / len(loader)
    elapsed_time = time.time() - start_time

    return acc, avg_loss, elapsed_time


def evaluate_with_shuffle(model, loader, device, criterion, shuffle_bio=False, shuffle_text=False):
    """
    [수정] shuffle_text 옵션 추가
    - shuffle_bio=True : 생체신호를 섞음 (텍스트 편향 확인)
    - shuffle_text=True: 텍스트를 섞음 (생체신호 편향 확인)
    """
    model.eval()
    
    total = 0
    correct = 0
    total_loss = 0.0
    
    all_true = []
    all_pred = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="TESTING", leave=False, dynamic_ncols=True, mininterval=0.1)
        
        for text_input, bio_input, label in pbar:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # -------------------------------------------------------
            # 🎲 Shuffling Logic
            # -------------------------------------------------------
            if shuffle_bio:
                # Bio만 섞기
                idx = torch.randperm(bio_input.size(0))
                bio_input = bio_input[idx]
            
            elif shuffle_text:
                # Text만 섞기 (딕셔너리 내부 모든 텐서를 같은 인덱스로 섞어야 함)
                idx = torch.randperm(label.size(0)) # 배치 사이즈만큼 랜덤 인덱스
                for k in text_input:
                    text_input[k] = text_input[k][idx]
            # -------------------------------------------------------

            logits = model(text_input, bio_input)
            loss = criterion(logits, label)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            all_true.extend(label.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    acc = correct / total
    avg_loss = total_loss / len(loader)
    
    return acc, avg_loss, all_true, all_pred