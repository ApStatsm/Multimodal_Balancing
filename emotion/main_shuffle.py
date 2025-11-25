from config import config
from utils import get_device
from dataset_multimodal import load_multimodal_data
from models.multimodal_e2e import MultimodalEndToEnd
from train import run_epoch, evaluate_with_shuffle 

import torch
import torch.nn as nn
# 🔥 [수정 1] train_test_split 추가
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 📊 혼동 행렬 그리기 도우미 함수
# ==========================================
def save_confusion_matrix(y_true, y_pred, title, filename):
    labels = ["Neutral", "Biased"]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # 메모리 해제
    print(f"   >> 💾 Saved: {filename}")

# ==========================================
# 🚀 메인 실행부
# ==========================================
def main():
    # 1. 초기 설정 및 데이터 로드
    tokenizer = None
    from kobert_tokenizer import KoBERTTokenizer
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    config["model"]["num_classes"] = 2
    print(f"🔧 Config Update: num_classes = {config['model']['num_classes']} (Binary: Neutral vs Biased)")

    full_loader, _, _ = load_multimodal_data(
        tokenizer=tokenizer,
        session_folder=config["paths"]["session_folder"],
        text_folder=config["paths"]["text_folder"],
        batch_size=config["training"]["batch_size"],
        max_len=config["model"]["max_len"]
    )
    full_dataset = full_loader.dataset

    # 전체 인덱스와 라벨 추출
    all_indices = np.arange(len(full_dataset))
    all_labels = np.array([full_dataset[i][2].item() for i in range(len(full_dataset))])
    
    print(f"\n📦 Total Data Samples: {len(all_indices)}")

    # ---------------------------------------------------------
    # 🔥 [수정 2] Test Set (20%) 영구 격리
    # ---------------------------------------------------------
    # dev_idx (80%): 학습(60%) + 검증(20%)에 사용
    # test_idx (20%): 최종 평가에만 사용 (LOCKED)
    dev_idx, test_idx = train_test_split(
        all_indices, 
        test_size=0.2, 
        stratify=all_labels, 
        random_state=42
    )
    
    # 격리된 Test용 Loader 생성
    test_subsampler = SubsetRandomSampler(test_idx)
    final_test_loader = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], sampler=test_subsampler)
    
    print(f"   🔹 Dev Set (For CV) : {len(dev_idx)} samples (80%)")
    print(f"   🔹 Final Test Set   : {len(test_idx)} samples (20%) - LOCKED 🔒")

    # ---------------------------------------------------------
    # 🔥 [수정 3] 4-Fold Cross Validation (남은 80%에 대해 수행)
    # ---------------------------------------------------------
    # Dev(80%)를 4등분하면 -> 1조각은 20%가 됨.
    # 즉, Train(3조각=60%) : Val(1조각=20%) 비율 완성
    n_splits = 4 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Dev 셋의 라벨만 따로 추출 (Stratified를 위해)
    dev_labels = all_labels[dev_idx]
    
    # 결과 저장소
    data_storage = {
        "normal": {"true": [], "pred": [], "acc": []},
        "bio_shuffled": {"true": [], "pred": [], "acc": []},
        "text_shuffled": {"true": [], "pred": [], "acc": []}
    }

    print(f"\n================ STARTING {n_splits}-FOLD CV ON DEV SET ================\n")
    print("   Target Split Ratio -> Train: 60% | Val: 20% | Test: 20%")

    # dev_idx를 가지고 K-Fold를 돕니다.
    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(skf.split(dev_idx, dev_labels)):
        print(f"\n----- Fold {fold_idx+1} / {n_splits} -----")
        
        # skf가 뱉는 건 dev_idx 내부의 '상대적 위치'이므로, '절대 인덱스'로 변환해줍니다.
        real_train_idx = dev_idx[inner_train_idx]
        real_val_idx   = dev_idx[inner_val_idx]
        
        train_subsampler = SubsetRandomSampler(real_train_idx)
        val_subsampler   = SubsetRandomSampler(real_val_idx) # Early Stopping용
        
        train_loader = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], sampler=train_subsampler)
        val_loader   = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], sampler=val_subsampler)
        
        model = MultimodalEndToEnd(config).to(device)
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
            # Train (60% 데이터 사용)
            train_acc, train_loss, _ = run_epoch(model, train_loader, optimizer, device, mode="train")
            # Val (20% 데이터 사용 - Early Stopping 체크)
            val_acc, val_loss, _ = run_epoch(model, val_loader, optimizer, device, mode="val")
            
            print(f"Ep {epoch+1:02d} | Train: {train_acc:.4f} ({train_loss:.4f}) | Val: {val_acc:.4f} ({val_loss:.4f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"   >> 🛑 Early Stopping (Best Val Loss: {best_val_loss:.4f})")
                    break
        
        # --- 최종 평가 (🔥 격리해둔 20% Test Set 사용) ---
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        # 1. Normal (정상)
        acc, _, t, p = evaluate_with_shuffle(model, final_test_loader, device, criterion, shuffle_bio=False, shuffle_text=False)
        data_storage["normal"]["acc"].append(acc)
        data_storage["normal"]["true"].extend(t)
        data_storage["normal"]["pred"].extend(p)

        # 2. Bio Shuffled (텍스트만 정상 -> 텍스트 의존도 확인)
        acc, _, t, p = evaluate_with_shuffle(model, final_test_loader, device, criterion, shuffle_bio=True, shuffle_text=False)
        data_storage["bio_shuffled"]["acc"].append(acc)
        data_storage["bio_shuffled"]["true"].extend(t)
        data_storage["bio_shuffled"]["pred"].extend(p)

        # 3. Text Shuffled (바이오만 정상 -> 바이오 의존도 확인)
        acc, _, t, p = evaluate_with_shuffle(model, final_test_loader, device, criterion, shuffle_bio=False, shuffle_text=True)
        data_storage["text_shuffled"]["acc"].append(acc)
        data_storage["text_shuffled"]["true"].extend(t)
        data_storage["text_shuffled"]["pred"].extend(p)

        print(f"   👉 [Final Test Set Result] Normal: {data_storage['normal']['acc'][-1]:.4f}")

    # ---------------------------------------------------------
    # 4. 최종 결과 요약
    # ---------------------------------------------------------
    print("\n================ FINAL SUMMARY (Test Set) ================\n")
    
    avg_norm = np.mean(data_storage["normal"]["acc"])
    avg_bio_shuf = np.mean(data_storage["bio_shuffled"]["acc"])
    avg_text_shuf = np.mean(data_storage["text_shuffled"]["acc"])
    
    print(f"1️⃣  Avg Normal Acc       : {avg_norm:.4f}")
    print("-" * 40)
    print(f"2️⃣  Avg Bio-Shuffled Acc : {avg_bio_shuf:.4f}")
    print(f"    -> Text Importance   : {avg_norm - avg_text_shuf:.4f} (Drop when Text is broken)")
    print("-" * 40)
    print(f"3️⃣  Avg Text-Shuffled Acc: {avg_text_shuf:.4f}")
    print(f"    -> Bio Importance    : {avg_norm - avg_bio_shuf:.4f} (Drop when Bio is broken)")
    
    print("\n[결론 해석]")
    drop_text_broken = avg_norm - avg_text_shuf
    drop_bio_broken = avg_norm - avg_bio_shuf
    
    if drop_text_broken > drop_bio_broken:
        print("🚨 모델은 **텍스트(Text)** 정보에 더 많이 의존하고 있습니다.")
    else:
        print("🚨 모델은 **생체신호(Bio)** 정보에 더 많이 의존하고 있습니다.")

    # ---------------------------------------------------------
    # 5. 3가지 Confusion Matrix 저장
    # ---------------------------------------------------------
    print("\nGenerating 3 Confusion Matrices (Based on Test Set)...")

    # (1) Normal CM
    save_confusion_matrix(
        data_storage["normal"]["true"], 
        data_storage["normal"]["pred"], 
        f"Confusion Matrix - Normal (Acc: {avg_norm:.4f})",
        "cm_1_normal.png"
    )

    # (2) Bio Shuffled CM
    save_confusion_matrix(
        data_storage["bio_shuffled"]["true"], 
        data_storage["bio_shuffled"]["pred"], 
        f"Confusion Matrix - Bio Shuffled (Text Only Effect)",
        "cm_2_bio_shuffled.png"
    )

    # (3) Text Shuffled CM
    save_confusion_matrix(
        data_storage["text_shuffled"]["true"], 
        data_storage["text_shuffled"]["pred"], 
        f"Confusion Matrix - Text Shuffled (Bio Only Effect)",
        "cm_3_text_shuffled.png"
    )

    print("\n================ DONE =================\n")

if __name__ == "__main__":
    main()