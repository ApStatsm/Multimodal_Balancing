from config import config
from utils import get_device
from dataset_multimodal import load_multimodal_data
from models.multimodal_e2e import MultimodalEndToEnd
# 🔥 train.py에서 학습 함수와 평가 함수를 가져옵니다.
from train import run_epoch, evaluate_with_shuffle 

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # ---------------------------------------------------------
    # 1. 초기 설정 및 데이터 로드
    # ---------------------------------------------------------
    tokenizer = None
    from kobert_tokenizer import KoBERTTokenizer
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    # [중요] 이진 분류 설정
    config["model"]["num_classes"] = 2
    print(f"🔧 Config Update: num_classes = {config['model']['num_classes']} (Binary: Neutral vs Biased)")

    # 전체 데이터 로드 (Split 없이 하나로)
    full_loader, _, _ = load_multimodal_data(
        tokenizer=tokenizer,
        session_folder=config["paths"]["session_folder"],
        text_folder=config["paths"]["text_folder"],
        batch_size=config["training"]["batch_size"],
        max_len=config["model"]["max_len"]
    )
    full_dataset = full_loader.dataset

    # K-Fold를 위한 라벨 추출
    all_labels = [full_dataset[i][2].item() for i in range(len(full_dataset))]
    all_labels = np.array(all_labels)

    # ---------------------------------------------------------
    # 2. 5-Fold Cross Validation 시작
    # ---------------------------------------------------------
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 전체 폴드의 예측 결과를 모을 리스트 (최종 Confusion Matrix용)
    global_true = []
    global_pred = []
    
    # 결과 요약용 리스트
    fold_results_normal_acc = []
    fold_results_shuffled_acc = []

    print(f"\n================ STARTING {n_splits}-FOLD CV ================\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n----- Fold {fold_idx+1} / {n_splits} -----")
        
        # 데이터 분할
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx) # Val이자 Test로 사용
        
        train_loader = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], sampler=train_subsampler)
        test_loader = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], sampler=val_subsampler)
        
        # 모델 초기화
        model = MultimodalEndToEnd(config).to(device)
        
        # Optimizer: Freeze된 레이어 제외하고 학습 (Fusion Layer 위주)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config["training"]["learning_rate"]
        )

        # --- 학습 루프 ---
        best_val_loss = float('inf')
        patience = 3
        counter = 0
        best_model_state = None
        
        for epoch in range(config["training"]["epochs"]):
            # Train
            train_acc, train_loss, train_time = run_epoch(model, train_loader, optimizer, device, mode="train")
            # Val
            val_acc, val_loss, val_time = run_epoch(model, test_loader, optimizer, device, mode="val")
            
            print(f"Ep {epoch+1:02d} | Train: {train_acc:.4f} ({train_loss:.4f}) | Val: {val_acc:.4f} ({val_loss:.4f})")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"   >> 🛑 Early Stopping (Best Val Loss: {best_val_loss:.4f})")
                    break
        
        # --- 테스트 & 평가 ---
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        # 1. Normal Test (정상)
        norm_acc, norm_loss, fold_true, fold_pred = evaluate_with_shuffle(
            model, test_loader, device, criterion, shuffle_bio=False
        )
        # 2. Shuffled Test (Bio 섞기 - 편향 분석용)
        shuf_acc, shuf_loss, _, _ = evaluate_with_shuffle(
            model, test_loader, device, criterion, shuffle_bio=True
        )
        
        fold_results_normal_acc.append(norm_acc)
        fold_results_shuffled_acc.append(shuf_acc)
        
        # Confusion Matrix를 위해 예측값 누적
        global_true.extend(fold_true)
        global_pred.extend(fold_pred)

        print(f"   👉 [Result] Normal Acc: {norm_acc:.4f} | Shuffled Acc: {shuf_acc:.4f}")
        print(f"   ⚠️  Gap: {norm_acc - shuf_acc:.4f}")

    # ---------------------------------------------------------
    # 3. 최종 결과 요약 및 시각화
    # ---------------------------------------------------------
    print("\n================ FINAL SUMMARY ================\n")
    
    avg_norm = np.mean(fold_results_normal_acc)
    avg_shuf = np.mean(fold_results_shuffled_acc)
    
    print(f"1️⃣  Avg Normal Accuracy   : {avg_norm:.4f}")
    print(f"2️⃣  Avg Shuffled Accuracy : {avg_shuf:.4f}")
    print(f"📉 Performance Drop (Gap)  : {avg_norm - avg_shuf:.4f}")
    
    if (avg_norm - avg_shuf) < 0.05:
        print("🚨 [해석] 성능 차이가 거의 없음 -> 텍스트(Text) 편향 의심")
    else:
        print("✅ [해석] 성능 하락 발생 -> 생체신호(Bio) 유의미하게 사용 중")

    # ---------------------------------------------------------
    # 4. Confusion Matrix 저장 (요청하신 스타일 적용)
    # ---------------------------------------------------------
    print("\nGenerating Confusion Matrix...")
    
    cm = confusion_matrix(global_true, global_pred)
    # 이진 분류 라벨
    labels = ["Neutral", "Biased"] 

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accumulated over {n_splits}-Folds)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print(f"💾 Confusion Matrix saved to 'confusion_matrix.png'")
    print("\n================ DONE =================\n")

if __name__ == "__main__":
    main()