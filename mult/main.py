import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup 
import os

# 우리가 만든 모듈들 import
from config import Config
from dataset import MultimodalDataset
from models import MultimodalFusion
from train import train_one_epoch, validate
from utils import set_seed, save_checkpoint

def main():
    # 1. 초기 설정
    set_seed(42)
    
    # Mac(MPS) 또는 CUDA 장치 설정
    device = torch.device(Config.DEVICE)
    print(f"🚀 프로젝트 시작: {Config.PROJECT_NAME}")
    print(f"💻 사용 장치: {device}") 

    # 저장 폴더 만들기
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    # 2. 토크나이저 & 데이터셋 로드
    print("\n[1/4] 데이터셋 준비 중...")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    # 전체 데이터 로드
    full_dataset = MultimodalDataset(Config.DATA_DIR, tokenizer, Config)
    
    # Train/Valid 분할
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"   - 전체 데이터: {total_size}개")
    print(f"   - 학습용(Train): {train_size}개")
    print(f"   - 검증용(Valid): {val_size}개")
    
    # 3. 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델 초기화
    print("\n[2/4] 모델 초기화 중 (KoBERT + LSTM Fusion)...")
    model = MultimodalFusion(Config).to(device)
    
    # ============================================================
    # [수정됨] PyTorch 내장 AdamW 사용
    # ============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss()
    
    # 6. 학습 루프 시작
    print("\n[3/4] 학습 시작!")
    best_val_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n📌 Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"   [Train] Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
        
        # 모델 저장
        if val_acc > best_val_acc:
            print(f"   🎉 성능 갱신! ({best_val_acc*100:.2f}% -> {val_acc*100:.2f}%) 모델 저장함.")
            best_val_acc = val_acc
            save_checkpoint(model, Config.MODEL_SAVE_PATH)
            
    print(f"\n[4/4] 모든 학습 완료. 최종 Best Acc: {best_val_acc*100:.2f}%")
    print(f"💾 모델 저장 위치: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()