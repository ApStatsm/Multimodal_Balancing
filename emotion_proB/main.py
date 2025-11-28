from config import config
from utils import get_device
from dataset_multimodal import load_data_frames, MultimodalDataset
from models.multimodal_e2e import MultimodalEndToEnd
from train import run_epoch, test_multimodal
import os
import torch
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer

# 📊 시각화 및 평가용 라이브러리
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_plots_and_report(scenario_name, labels, preds, probs):
    """
    결과 출력 및 시각화 저장 함수 (emotion/result 폴더에 저장)
    """
    output_dir = os.path.join("emotion", "result")
    os.makedirs(output_dir, exist_ok=True)  
    
    # 🔥 [수정] 5진 분류 라벨 이름 정의
    target_names = ["Neutral", "Surprise", "Angry", "Sad", "Happy"]

    print(f"\n>> Classification Report ({scenario_name}):")
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Neutral", "Biased"], yticklabels=["Neutral", "Biased"])
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

    # 1. 데이터 로드 (Train 80% : Test 20%)
    # load_data_frames 내부에서 8:2로 나뉘어 나옵니다.
    train_df, test_df = load_data_frames(config["paths"]["session_folder"])
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    print(f"\n[Data Split Info]")
    print(f"Train Set: {len(train_df)} samples")
    print(f"Test Set : {len(test_df)} samples")

    # 2. DataLoader 생성 (Train만 만듦, Test는 나중에)
    train_ds = MultimodalDataset(train_df, config["paths"]["text_folder"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)

    # 3. 모델 및 옵티마이저 초기화
    model = MultimodalEndToEnd(config).to(device)
    
    # KoBERT 제외 전체 파라미터 학습
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["training"]["learning_rate"]
    )

    print(f"\n[Start Training for {config['training']['epochs']} Epochs]")

    # 4. Training Loop (검증 없이 학습만 진행)
    for epoch in range(config["training"]["epochs"]):
        # Train 모드 실행
        t_acc, t_loss = run_epoch(model, train_loader, optimizer, device, "train")
        
        # 로그 출력 (Val 없음)
        print(f"Ep {epoch+1:02d} | Train Acc: {t_acc:.3f} | Train Loss: {t_loss:.4f}")

    # ================= FINAL TEST =================
    print("\n\n================ FINAL EVALUATION ================")
    
    # 학습이 끝난 모델 그대로 사용
    test_ds = MultimodalDataset(test_df, config["paths"]["text_folder"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # 3가지 시나리오 정의
    scenarios = [
        ("1_Normal_Test", "none"),          # 정상
        ("2_Bio_Only_(Text_Masked)", "text"), # 텍스트 마스킹
        ("3_Text_Only_(Bio_Masked)", "bio")   # Bio 마스킹
    ]

    for name, mode in scenarios:
        print(f"\n\n🔶 Running Scenario: {name}")
        
        # 1. 테스트 실행
        acc, loss, labels, preds, probs = test_multimodal(model, test_loader, device, shuffle_mode=mode)
        
        # 2. Acc / Loss 출력
        print(f"▶ Test Acc : {acc:.4f}")
        print(f"▶ Test Loss: {loss:.4f}")
        
        # 3. Report, AUC, CM 저장
        save_plots_and_report(name, labels, preds, probs)

    print("\n✅ All experiments done.")

if __name__ == "__main__":
    main()