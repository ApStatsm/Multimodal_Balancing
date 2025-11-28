# main.py
import torch
import torch.nn as nn
from transformers import logging
logging.set_verbosity_error()
from kobert_tokenizer import KoBERTTokenizer
from dataset import load_data_from_folders
from model import KoBERTClassifier
from train import train_one_epoch, evaluate
from inference import show_misclassified
from utils import get_device
import time
# 1️⃣ [추가] roc_curve, auc 임포트
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 경로 설정
    csv_path = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data"
    text_folder = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/KEMDy19_v1_3/wav"

    # 설정
    num_classes = 2      # neutral vs biased
    epochs = 10
    batch_size = 32
    lr = 5e-5

    device = get_device()
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    # 데이터 로드
    train_loader, test_loader = load_data_from_folders(
        tokenizer=tokenizer,
        csv_path=csv_path,
        text_folder=text_folder,
        batch_size=batch_size
    )
    
    model = KoBERTClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Training Start (Epochs: {epochs}, Device: {device})\n")
    
    for epoch in range(epochs):
        # ... (학습 및 에포크별 출력 로직 기존과 동일) ...
        start_time = time.time()

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 검증 (AUC 받아오기)
        test_loss, test_acc, test_auc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        # 에포크 결과 출력 (AUC 추가)
        print(f"-"*55)
        print(f"📄 [Epoch {epoch + 1}/{epochs}] Results")
        print(f"   - Train Loss : {train_loss:.4f} | Acc : {train_acc:.4f}")
        print(f"   - Test  Loss : {test_loss:.4f} | Acc : {test_acc:.4f} | AUC : {test_auc:.4f}")
        print(f"   - Time       : {epoch_time:.2f} sec")
        print(f"-"*55 + "\n")

    # --- 최종 평가 ---
    print("\n🏁 FINAL Model Evaluation (Test Set) ---")

    # 2️⃣ evaluate 함수에서 확률(probs)도 함께 반환받음
    test_loss, test_acc, test_auc, preds, trues, probs = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Final Test Loss     : {test_loss:.4f}")
    print(f"Final Test Accuracy : {test_acc:.4f}")
    print(f"Final Test AUC      : {test_auc:.4f}")

    id2label = {0: "neutral", 1: "biased"}
    y_test = [id2label[t] for t in trues]
    y_pred_test = [id2label[p] for p in preds]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, digits=4, zero_division=0))

    # --- Confusion Matrix 저장 ---
    labels_order = ["neutral", "biased"]
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (AUC: {test_auc:.4f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix_binary_auc.png", dpi=300)
    plt.close()
    print("💾 Confusion Matrix saved to 'confusion_matrix_binary_auc.png'")

    # 3️⃣ --- ROC Curve 그래프 그리기 및 저장 ---
    # FPR, TPR, 임계값 계산
    fpr, tpr, _ = roc_curve(trues, probs)
    # AUC 계산 (이미 test_auc로 받았지만, 그래프 범례용으로 다시 계산하거나 그대로 사용 가능)
    roc_auc = auc(fpr, tpr) 

    plt.figure(figsize=(8, 6))
    # 짙은 파란색 점선으로 랜덤 추측선 (대각선) 그리기
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # 주황색 실선으로 ROC 곡선 그리기
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - Neutral vs Biased (KoBERT)') # 제목 설정
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray') # 그리드 추가
    plt.tight_layout()
    
    # 이미지 파일로 저장
    roc_image_path = "roc_curve_binary.png"
    plt.savefig(roc_image_path, dpi=300)
    plt.close()
    print(f"💾 ROC Curve saved to '{roc_image_path}'")
    # -------------------------------------

    show_misclassified(model, test_loader, device, label_map=id2label)

if __name__ == "__main__":
    main()