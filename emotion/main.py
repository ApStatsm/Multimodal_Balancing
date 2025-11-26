from config import config
from utils import get_device
from dataset_multimodal import load_data_frames, MultimodalDataset
from models.multimodal_e2e import MultimodalEndToEnd
from train import run_epoch, test_multimodal

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from kobert_tokenizer import KoBERTTokenizer
import copy

# 📊 시각화 및 평가용 라이브러리
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_plots_and_report(scenario_name, labels, preds, probs):
    """
    결과 출력 및 시각화 저장 함수
    """
    print(f"\n>> Classification Report ({scenario_name}):")
    # target_names: 0=Neutral, 1=Biased
    print(classification_report(labels, preds, target_names=["Neutral", "Biased"], digits=4))

    # 1. Confusion Matrix 저장
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Neutral", "Biased"], yticklabels=["Neutral", "Biased"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {scenario_name}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{scenario_name}.png")
    plt.close()

    # 2. AUC 시각화 저장
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {scenario_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"roc_curve_{scenario_name}.png")
    plt.close()
    
    print(f"✅ Saved plots for {scenario_name}")


def main():
    device = get_device()
    print(f"Running on Device: {device}")

    # 데이터 로드
    train_df, test_df = load_data_frames(config["paths"]["session_folder"])
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    # K-Fold 설정
    skf = StratifiedKFold(n_splits=config["training"]["k_folds"], shuffle=True, random_state=42)
    
    best_val_acc = 0.0
    best_model_state = None

    print(f"\n[Start {config['training']['k_folds']}-Fold Cross Validation]")

    # K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
        print(f"\n=== Fold {fold+1} ===")
        
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        train_ds = MultimodalDataset(fold_train, config["paths"]["text_folder"], tokenizer)
        val_ds = MultimodalDataset(fold_val, config["paths"]["text_folder"], tokenizer)
        
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

        model = MultimodalEndToEnd(config).to(device)
        
        # Optimizer: KoBERT 제외 전체 파라미터 학습
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config["training"]["learning_rate"]
        )

        for epoch in range(config["training"]["epochs"]):
            t_acc, t_loss = run_epoch(model, train_loader, optimizer, device, "train")
            v_acc, v_loss = run_epoch(model, val_loader, optimizer, device, "val")
            print(f"Ep {epoch+1:02d} | Train: {t_acc:.3f} | Val: {v_acc:.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"⭐️ New Best Model (Val: {best_val_acc:.3f})")

    # ================= FINAL TEST =================
    print("\n\n================ FINAL EVALUATION ================")
    
    final_model = MultimodalEndToEnd(config).to(device)
    final_model.load_state_dict(best_model_state)
    
    test_ds = MultimodalDataset(test_df, config["paths"]["text_folder"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # 3가지 시나리오 정의
    scenarios = [
        ("1_Normal_Test", "none"),          # 정상
        ("2_Bio_Only_(Text_Masked)", "text"), # 텍스트 마스킹 (Bio 의존도 확인)
        ("3_Text_Only_(Bio_Masked)", "bio")   # Bio 마스킹 (Text 의존도 확인)
    ]

    for name, mode in scenarios:
        print(f"\n\n🔶 Running Scenario: {name}")
        
        # 1. 테스트 실행
        acc, loss, labels, preds, probs = test_multimodal(final_model, test_loader, device, shuffle_mode=mode)
        
        # 2. Acc / Loss 출력
        print(f"▶ Test Acc : {acc:.4f}")
        print(f"▶ Test Loss: {loss:.4f}")
        
        # 3. Report, AUC, CM 저장
        save_plots_and_report(name, labels, preds, probs)

    print("\n✅ All experiments done.")

if __name__ == "__main__":
    main()