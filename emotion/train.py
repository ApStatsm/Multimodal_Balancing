import torch
import torch.nn as nn
from tqdm import tqdm
import time

def run_epoch(model, loader, optimizer, device, mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total, correct = 0, 0
    total_loss = 0
    ce = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(mode == "train"):
        for text_input, bio_input, label in loader: # tqdm 제거 (K-Fold 출력 깔끔하게)
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            if mode == "train":
                optimizer.zero_grad()

            logits = model(text_input, bio_input)
            loss = ce(logits, label)

            if mode == "train":
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return acc, avg_loss

# 🔥 [수정 4] 셔플링 테스트 함수
def test_multimodal(model, loader, device, shuffle_mode="none"):
    """
    shuffle_mode: "none", "text", "bio"
    """
    model.eval()
    total, correct = 0, 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text_input, bio_input, label in loader:
            
            # GPU 이동
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # 🎲 셔플 로직 적용 (배치 내 셔플)
            idx = torch.randperm(bio_input.size(0)).to(device)

            if shuffle_mode == "text":
                # 텍스트만 섞음 (Input ID, Mask 등 모두 섞어야 함)
                for k in text_input:
                    text_input[k] = text_input[k][idx]
            
            elif shuffle_mode == "bio":
                # 바이오만 섞음
                bio_input = bio_input[idx]

            logits = model(text_input, bio_input)
            pred = logits.argmax(dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    acc = correct / total
    return acc, all_labels, all_preds