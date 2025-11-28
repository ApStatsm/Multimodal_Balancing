# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# 🔥 [추가] config 임포트
from config import config 

def run_epoch(model, loader, optimizer, device, mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    # 🔥 [추가] Loss 가중치 불러오기
    ALPHA = config["balancing"]["alpha"]
    BETA = config["balancing"]["beta"]
    LAMBDA = config["balancing"]["lambda"]
    
    total, correct = 0, 0
    total_loss = 0
    ce = nn.CrossEntropyLoss()
    # 🔥 [추가] 일관성 손실을 위한 L2 거리 (MSE)
    mse = nn.MSELoss() 

    loader_pbar = tqdm(loader, desc=f"{mode.upper()}", leave=False)

    with torch.set_grad_enabled(mode == "train"):
        for text_input, bio_input, label in loader_pbar:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            if mode == "train":
                optimizer.zero_grad()

            # --- 🔥 [STEP 1] Full Input으로 학습 (L_CE, L_text, L_bio, h_full 계산) ---
            # MultimodalEndToEnd의 4가지 반환 값 수신
            final_logits, h_full, aux_text_logits, aux_bio_logits = model(text_input, bio_input)

            # 1. L_CE (Main Classification Loss)
            loss_ce = ce(final_logits, label)

            # 2. L_Auxiliary (Unimodal Loss)
            loss_text = ce(aux_text_logits, label)
            loss_bio = ce(aux_bio_logits, label)
            
            # --- 🔥 [STEP 2] Masked Input으로 L_cons 계산 ---
            loss_cons = torch.tensor(0.0).to(device) # 0.0 대신 텐서로 초기화
            
            if mode == "train" and LAMBDA > 0:
                # 2-1. 텍스트 마스킹 (Bio Only 효과)
                # 텍스트 input의 값을 셔플하여 데이터 간의 의미 없는 조합을 만듦
                # (LANISTR의 마스킹 방식 중 하나인 Cross-Modality Shuffling 적용)
                text_input_masked = {k: v.clone() for k, v in text_input.items()}
                idx = torch.randperm(bio_input.size(0)).to(device)
                for k in text_input_masked:
                    text_input_masked[k] = text_input_masked[k][idx] 
                
                # 마스킹된 텍스트와 원본 바이오로 forward 실행. (로짓은 무시)
                _, h_text_masked, _, _ = model(text_input_masked, bio_input)
                loss_cons += mse(h_full, h_text_masked) # 일관성 손실 1

                # 2-2. 바이오 마스킹 (Text Only 효과)
                # 바이오 input의 값을 셔플
                bio_input_masked = bio_input.clone()
                idx = torch.randperm(bio_input.size(0)).to(device)
                bio_input_masked = bio_input_masked[idx]
                
                # 원본 텍스트와 마스킹된 바이오로 forward 실행
                _, h_bio_masked, _, _ = model(text_input, bio_input_masked)
                loss_cons += mse(h_full, h_bio_masked) # 일관성 손실 2


            # --- 🔥 [STEP 3] 최종 Total Loss 합산 ---
            loss = loss_ce + (ALPHA * loss_text + BETA * loss_bio) + (LAMBDA * loss_cons)
            
            if mode == "train":
                loss.backward()
                optimizer.step()

            pred = final_logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item() # 최종 합산 Loss를 기록
            
            # 진행 바 옆에 실시간 Loss/Acc 표시
            current_acc = correct / total
            loader_pbar.set_postfix({'loss': loss.item(), 'acc': current_acc})

    return correct / total, total_loss / len(loader)


def test_multimodal(model, loader, device, shuffle_mode="none"):
    """
    테스트 모드: shuffle_mode에 따라 다른 모달리티를 셔플하여 편향 분석을 수행합니다.
    shuffle_mode: "none", "text", "bio"
    Returns: acc, loss, labels, preds, probs (for AUC)
    """
    model.eval()
    total_loss = 0
    ce = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_probs = []

    # 테스트 단계에서도 진행 상황 보기
    loader_pbar = tqdm(loader, desc=f"TEST({shuffle_mode})", leave=True)

    with torch.no_grad():
        for text_input, bio_input, label in loader_pbar:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # 🎲 셔플 로직
            idx = torch.randperm(bio_input.size(0)).to(device)
            if shuffle_mode == "text":
                for k in text_input:
                    text_input[k] = text_input[k][idx]
            elif shuffle_mode == "bio":
                bio_input = bio_input[idx]

            # 🔥 [수정] model(text_input, bio_input)의 4가지 반환값 중 최종 로짓만 사용
            # final_logits, h_full, aux_text_logits, aux_bio_logits = model(...)
            final_logits, _, _, _ = model(text_input, bio_input)
            
            loss = ce(final_logits, label)
            total_loss += loss.item()

            # 🔥 [수정] 다중 분류이므로 [:, 1] 인덱싱 제거
            # 전체 확률 분포를 저장 (나중에 분석할 때 필요하면 사용)
            probs = F.softmax(final_logits, dim=1)
            pred = final_logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            # 다중 분류 확률값 저장 (N, 5)
            all_probs.extend(probs.cpu().tolist())

            loader_pbar.set_postfix({'loss': loss.item()})


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    return acc, total_loss / len(loader), all_labels, all_preds, all_probs