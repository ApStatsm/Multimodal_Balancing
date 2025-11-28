# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# train_one_epoch은 기존과 동일
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, trues = 0, [], []
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds += outputs.argmax(1).detach().cpu().tolist()
        trues += labels.cpu().tolist()
        progress_bar.set_postfix(loss=loss.item())

    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc

# ✅ [수정 대상] evaluate 함수
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, trues, probs = 0, [], [], []
    
    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds += outputs.argmax(1).cpu().tolist()
            trues += labels.cpu().tolist()
            
            # 확률값 추출
            probs += F.softmax(outputs, dim=1)[:, 1].cpu().tolist()

    acc = accuracy_score(trues, preds)
    
    try:
        auc = roc_auc_score(trues, probs)
    except ValueError:
        auc = 0.0

    # 🚨 수정된 부분: 마지막에 probs를 추가하여 총 6개를 반환합니다.
    # 맨 마지막에 , probs 를 꼭 추가해주세요!
    return total_loss / len(loader), acc, auc, preds, trues, probs