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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from inference import show_misclassified
from utils import get_device

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
def main():
    #  데이터 경로 설정
    csv_path = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data"      # 여러 CSV가 들어있는 폴더
    text_folder = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/KEMDy19_v1_3/wav"

    num_classes = 5
    epochs = 10
    batch_size = 32
    lr = 5e-5

    device = get_device()
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    #  데이터 로드
    train_loader, test_loader = load_data_from_folders(
        tokenizer=tokenizer,
        csv_path=csv_path,
        text_folder=text_folder,
        batch_size=batch_size
    )
    

    model = KoBERTClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n Training Start\n")
    for epoch in range(epochs):
        start_time = time.time()

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 매 epoch마다 test 성능 체크 (원하면 나중에 빼도 됨)
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"[Epoch {epoch + 1}]")
        print(f" Train  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f" Test   Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f" ⏱️ Time per epoch: {epoch_time:.2f} sec\n")

    #  --- 학습 종료 후, 최종 평가 ---
    #  --- 학습 종료 후, 최종 평가 ---
    print("\n--- FINAL Model Evaluation (Test Set) ---")

    # test set에서 최종 loss / acc / 예측 / 정답 모두 계산
    test_loss, test_acc, preds, trues = evaluate(
        model, test_loader, criterion, device
    )

    # 1) Acc / Loss 출력
    print(f"Final Test Loss     : {test_loss:.4f}")
    print(f"Final Test Accuracy : {test_acc:.4f}")

    # 감정 인덱스 ↔ 이름 매핑
    id2label = {
        0: "neutral",
        1: "surprise",
        2: "angry",
        3: "sad",
        4: "happy"
    }

    # 숫자 레이블을 문자열로 변환
    y_test = [id2label[t] for t in trues]
    y_pred_test = [id2label[p] for p in preds]

    # 2) Classification Report 출력
    print("\nFinal Classification Report (Test Set):")
    print(classification_report(
        y_test,
        y_pred_test,
        digits=4,  # 소수점 4자리
        zero_division=0  # 분모가 0인 경우 0으로 처리
    ))

    # 3) Confusion Matrix (숫자 + 이미지 저장)
    labels_order = list(id2label.values())  # ['happy','surprise','angry','neutral','sad']
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)

    print("\nConfusion Matrix (raw counts, Test Set):")
    print(cm)

    # 이미지로 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels_order,
                yticklabels=labels_order)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (KEMDy20 Test)")
    plt.tight_layout()

    # 이미지 파일로 저장 (원하는 이름으로 변경 가능)
    plt.savefig("confusion_matrix_kemdy20.png", dpi=300)
    plt.close()

    # 오분류 샘플 저장/출력
    show_misclassified(model, test_loader, device)


if __name__ == "__main__":
    main()
