import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.modules.loss import _WeightedLoss # 🔥 Loss 가중치 사용을 위해 임포트
from sklearn.metrics import accuracy_score

# 🔥 Custom Weighted Cross Entropy Loss
class WeightedCrossEntropyLoss(_WeightedLoss):
    def forward(self, input, target, weight=None):
        # reduction='none'으로 설정하여 배치 크기만큼의 Loss 벡터를 얻습니다.
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        if weight is not None:
            # Loss 벡터에 샘플별 가중치를 곱합니다.
            ce_loss = ce_loss * weight

        # 평균 Loss를 반환합니다.
        return ce_loss.mean()


# run_epoch 함수 수정: weight를 받고, WeightedCrossEntropyLoss 사용
def run_epoch(model, loader, optimizer, device, mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total, correct = 0, 0
    total_loss = 0
    # 🔥 nn.CrossEntropyLoss() 대신 Custom Loss 사용
    ce = WeightedCrossEntropyLoss()

    loader_pbar = tqdm(loader, desc=f"{mode.upper()}", leave=False)

    with torch.set_grad_enabled(mode == "train") and torch.autograd.set_detect_anomaly(False):
        # 🔥 weight 변수 추가
        for text_input, bio_input, label, weight in loader_pbar:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)
            # 🔥 weight를 장치로 이동
            weight = weight.to(device) 

            if mode == "train":
                optimizer.zero_grad()

            logits = model(text_input, bio_input)
            # 🔥 loss 계산 시 weight 전달
            loss = ce(logits, label, weight=weight) 

            if mode == "train":
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()
            
            current_acc = correct / total
            loader_pbar.set_postfix({'loss': loss.item(), 'acc': current_acc})

    return total_loss / len(loader), current_acc


# test_multimodal 함수 수정: 모달리티 Zero-out 로직 구현
def test_multimodal(model, loader, device, shuffle_mode="none"):
    """
    모달리티 Zero-out을 위한 test 함수
    shuffle_mode: "none" (정상), "text_zeroout" (Bio-Only), "bio_zeroout" (Text-Only)
    Returns: acc, loss, labels, preds, probs
    """
    model.eval()
    total_loss = 0
    ce = nn.CrossEntropyLoss() # 테스트 시에는 일반 CrossEntropyLoss 사용
    
    all_preds = []
    all_labels = []
    all_probs = []

    loader_pbar = tqdm(loader, desc=f"TEST({shuffle_mode})", leave=True)

    with torch.no_grad():
        # 🔥 test 시에는 weight가 필요 없으나, Dataset 반환 형식에 맞춰 받아옵니다.
        for text_input, bio_input, label, _ in loader_pbar: 
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # 🔥 Zero-out 로직 구현
            if shuffle_mode == "text_zeroout":
                # 텍스트 정보 마스킹: input_ids와 attention_mask를 0으로 설정
                # KoBERT가 frozen 상태이므로, input이 0이면 특정 고정된 벡터가 나올 것임
                # (또는 단순히 KoBERT의 결과인 text_feat를 0으로 설정하는 것도 가능하나,
                #  여기서는 input을 0으로 만들어 일관성을 유지합니다.)
                for k in text_input:
                     text_input[k].zero_() # 모든 원소를 0으로 

            elif shuffle_mode == "bio_zeroout":
                # Bio 정보 마스킹: Bio input 벡터를 0으로 설정
                bio_input.zero_() # 모든 원소를 0으로

            logits = model(text_input, bio_input)
            loss = ce(logits, label)
            total_loss += loss.item()

            # AUC 계산용 확률값 (이진 분류 시 1번 클래스 확률)
            probs = F.softmax(logits, dim=1)[:, 1] 
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    
    return acc, avg_loss, all_labels, all_preds, all_probs