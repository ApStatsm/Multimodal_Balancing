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
from sklearn.metrics import classification_report, accuracy_score

def main():
    #  데이터 경로 설정
    csv_path = r'/Users/apstat/Desktop/02_연구/01_멀티모달 밸런싱/데이터'     # 여러 CSV가 들어있는 폴더
    text_folder = r'/Users/apstat/Desktop/02_연구/01_멀티모달 밸런싱/KEMDy20_v1_2/wav'

    num_classes = 7
    epochs = 5
    batch_size = 32
    lr = 2e-5

    device = get_device()
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    #  데이터 로드
    train_loader, val_loader, test_loader = load_data_from_folders(
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

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"[Epoch {epoch+1}]")
        print(f" Train  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f" Val    Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f" Test   Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f" ⏱️ Time per epoch: {epoch_time:.2f} sec\n")

    #  --- 학습 종료 후, 최종 평가 ---
    print("\n--- FINAL Model Evaluation (Test Set) ---")

    # 모델의 예측/정답 수집
    _, _, preds, trues = evaluate(model, test_loader, criterion, device)

    # 정확도 계산
    accuracy_test = accuracy_score(trues, preds)
    print(f"Test Accuracy: {accuracy_test:.4f}")

    # 감정 인덱스 ↔ 이름 매핑
    id2label = {
        0: "happy",
        1: "surprise",
        2: "angry",
        3: "neutral",
        4: "disgust",
        5: "fear",
        6: "sad"
    }

    # 문자열 레이블로 변환
    y_test = [id2label[t] for t in trues]
    y_pred_test = [id2label[p] for p in preds]

    #  Classification Report 출력
    print("\nFinal Classification Report (Test Set):")
    print(classification_report(
        y_test, y_pred_test,
        digits=4,
        zero_division=0
    ))

    #  오분류 샘플 CSV 저장
    show_misclassified(model, test_loader, device)


if __name__ == "__main__":
    main()
