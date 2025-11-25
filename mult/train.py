import torch
from tqdm import tqdm # 진행률 표시 라이브러리 (pip install tqdm 필요)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1 에폭(Epoch) 학습 함수"""
    model.train() # 모델을 학습 모드로 전환 (Dropout 켜기)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 진행률 바 생성
    progress_bar = tqdm(dataloader, desc="[Train]", leave=False)
    
    for batch in progress_bar:
        # 1. 데이터를 맥 GPU(MPS)로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        signal_input = batch['signal_input'].to(device)
        labels = batch['label'].to(device)
        
        # 2. 그래디언트 초기화
        optimizer.zero_grad()
        
        # 3. Forward (예측)
        outputs = model(input_ids, attention_mask, signal_input)
        
        # 4. Loss 계산
        loss = criterion(outputs, labels)
        
        # 5. Backward (역전파) & Update (가중치 갱신)
        loss.backward()
        optimizer.step()
        
        # 6. 통계 계산 (정확도 등)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 진행률 바에 현재 Loss 표시
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """검증(Validation) 함수"""
    model.eval() # 모델을 평가 모드로 전환 (Dropout 끄기)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 평가 때는 미분 계산 안 함 (속도 UP, 메모리 절약)
        for batch in tqdm(dataloader, desc="[Valid]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            signal_input = batch['signal_input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, signal_input)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy