from config import config
from utils import get_device
from dataset_multimodal import load_data_frames, MultimodalDataset
from models.multimodal_e2e import MultimodalEndToEnd
from train import run_epoch, test_multimodal
import os
import torch
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer

# 📊 시각화
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 🔥 [수정] 순서 변경 (0: Neutral, 1: Surprise, 2: Angry, 3: Sad, 4: Happy)
TARGET_NAMES = ["Neutral", "Surprise", "Angry", "Sad", "Happy"]

def save_plots_and_report(scenario_name, labels, preds):
    output_dir = os.path.join("emotion", "result")
    os.makedirs(output_dir, exist_ok=True)  
    
    print(f"\n>> Classification Report ({scenario_name}):")
    # labels에 없는 클래스가 있을 경우를 대비해 labels 매개변수 명시
    print(classification_report(labels, preds, target_names=TARGET_NAMES, digits=4, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {scenario_name}')
    plt.tight_layout()
    
    save_path_cm = os.path.join(output_dir, f"confusion_matrix_{scenario_name}.png")
    plt.savefig(save_path_cm)
    plt.close()
    
    print(f"✅ Saved plots to '{output_dir}' for {scenario_name}")

def main():
    device = get_device()
    print(f"Running on Device: {device}")

    # 1. 데이터 로드 (5 Class)
    train_df, test_df = load_data_frames(config["paths"]["session_folder"])
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    print(f"\n[Data Split Info]")
    print(f"Train Set: {len(train_df)} samples")
    print(f"Test Set : {len(test_df)} samples")

    # 2. DataLoader
    train_ds = MultimodalDataset(train_df, config["paths"]["text_folder"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)

    # 3. 모델 초기화
    model = MultimodalEndToEnd(config).to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["training"]["learning_rate"]
    )

    print(f"\n[Start Training for {config['training']['epochs']} Epochs]")

    # 4. Training (Train Only)
    for epoch in range(config["training"]["epochs"]):
        t_acc, t_loss = run_epoch(model, train_loader, optimizer, device, "train")
        print(f"Ep {epoch+1:02d} | Train Acc: {t_acc:.3f} | Train Loss: {t_loss:.4f}")

    # ================= FINAL TEST =================
    print("\n\n================ FINAL EVALUATION ================")
    
    test_ds = MultimodalDataset(test_df, config["paths"]["text_folder"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    scenarios = [
        ("1_Normal_Test", "none"),
        ("2_Bio_Only_(Text_Masked)", "text"),
        ("3_Text_Only_(Bio_Masked)", "bio")
    ]

    for name, mode in scenarios:
        print(f"\n\n🔶 Running Scenario: {name}")
        
        # test_multimodal에서 probs 반환 제거함
        acc, loss, labels, preds = test_multimodal(model, test_loader, device, shuffle_mode=mode)
        
        print(f"▶ Test Acc : {acc:.4f}")
        print(f"▶ Test Loss: {loss:.4f}")
        
        save_plots_and_report(name, labels, preds)

    print("\n✅ All experiments done.")

if __name__ == "__main__":
    main()