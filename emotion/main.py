from config import config
from utils import get_device
from dataset_multimodal import load_multimodal_data
from models.multimodal_e2e import MultimodalEndToEnd
from train import run_epoch
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    tokenizer = None
    from kobert_tokenizer import KoBERTTokenizer
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    device = get_device()

    train_loader, val_loader, test_loader = load_multimodal_data(
        tokenizer=tokenizer,
        session_folder=config["paths"]["session_folder"],
        text_folder=config["paths"]["text_folder"],
        batch_size=config["training"]["batch_size"],
        max_len=config["model"]["max_len"]
    )

    model = MultimodalEndToEnd(config).to(device)

    # 🔥 Frozen 기반 바닐라 멀티모달이므로 fusion만 학습
    optimizer = torch.optim.Adam(
        model.fusion.parameters(),
        lr=config["training"]["learning_rate"]
    )

    print("\n================ TRAINING START =================\n")

    for epoch in range(config["training"]["epochs"]):
        print(f"\n----- Epoch {epoch+1} / {config['training']['epochs']} -----")

        # TRAIN
        train_acc, train_loss, train_time = run_epoch(
            model, train_loader, optimizer, device, mode="train"
        )
        print(f"Train Acc:  {train_acc:.4f} | Loss: {train_loss:.4f} | Time: {train_time:.2f}s")

        # VAL
        val_acc, val_loss, val_time = run_epoch(
            model, val_loader, optimizer, device, mode="val"
        )
        print(f"Val   Acc:  {val_acc:.4f} | Loss: {val_loss:.4f} | Time: {val_time:.2f}s")

    print("\n================ TESTING START =================\n")

    test_acc, test_loss, test_time = run_epoch(
        model, test_loader, optimizer, device, mode="test"
    )

    print(f"TEST Acc:  {test_acc:.4f} | Loss: {test_loss:.4f} | Time: {test_time:.2f}s")
    # =======================
    # Confusion Matrix 저장
    # =======================
    all_true = []
    all_pred = []

    model.eval()
    with torch.no_grad():
        for text_input, bio_input, label in test_loader:

            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            logits = model(text_input, bio_input)
            pred = logits.argmax(dim=1)

            all_true.extend(label.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    # Confusion Matrix 생성
    cm = confusion_matrix(all_true, all_pred)
    labels = ["happy", "surprise", "angry", "neutral", "disgust", "fear", "sad"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\n================ DONE =================\n")


if __name__ == "__main__":
    main()
