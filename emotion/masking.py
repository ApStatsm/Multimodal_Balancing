import torch
import os
import numpy as np
import matplotlib.pyplot as plt  # 👈 시각화용
import seaborn as sns            # 👈 히트맵용
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # 👈 추가됨

# 기존 파일들에서 필요한 모듈 임포트
from config import config
from utils import get_device
from dataset_multimodal import load_multimodal_data
from models.multimodal_e2e import MultimodalEndToEnd
from kobert_tokenizer import KoBERTTokenizer

# ==========================================
# 📊 Confusion Matrix 시각화 함수
# ==========================================
def plot_confusion_matrix(y_true, y_pred, mode_name, save_dir="results"):
    """
    Confusion Matrix를 계산하고 이미지로 저장합니다.
    """
    # 결과 저장 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {mode_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 파일명에 특수문자나 공백이 있으면 처리
    safe_name = mode_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(save_dir, f"cm_{safe_name}.png")
    
    plt.savefig(save_path)
    plt.close() # 메모리 해제
    print(f"   >> 💾 Confusion Matrix saved to: {save_path}")

# ==========================================
# 🔍 모달리티 불균형 분석 함수 (마스킹 테스트)
# ==========================================
def evaluate_modality_bias(model, test_loader, device, tokenizer):
    model.eval()
    
    # 3가지 모드
    modes = ["Original (Both)", "Text Only (Bio Masked)", "Bio Only (Text Masked)"]
    results = {}

    print("\n" + "="*20 + " 🔍 MODALITY BIAS ANALYSIS " + "="*20)

    for mode in modes:
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for text_input, bio_input, label in test_loader:
                
                # 데이터 장치로 이동
                for k in text_input:
                    text_input[k] = text_input[k].to(device)
                bio_input = bio_input.to(device)
                label_cpu = label.numpy() 

                # -------------------------------------------
                # 🎭 마스킹(Masking) 로직
                # -------------------------------------------
                if mode == "Text Only (Bio Masked)":
                    bio_input = torch.zeros_like(bio_input).to(device)

                elif mode == "Bio Only (Text Masked)":
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
                    text_input["input_ids"].fill_(pad_id)
                    text_input["attention_mask"].fill_(0)
                    text_input["token_type_ids"].fill_(0)
                # -------------------------------------------

                logits = model(text_input, bio_input)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_true.extend(label_cpu)
                all_pred.extend(preds)

        # -------------------------------------------
        # 📊 결과 집계 및 출력
        # -------------------------------------------
        acc = accuracy_score(all_true, all_pred)
        results[mode] = acc
        
        print(f"\n📌 [{mode}] Test Results")
        print(f"   >> Accuracy: {acc:.4f}")
        
        # 1. Classification Report 출력
        print("-" * 55)
        print(classification_report(all_true, all_pred, digits=4, zero_division=0))
        
        # 2. Confusion Matrix 시각화 및 저장
        plot_confusion_matrix(all_true, all_pred, mode)
        
        print("=" * 55)

    # -------------------------------------------
    # 📉 결과 해석 출력
    # -------------------------------------------
    orig_acc = results["Original (Both)"]
    text_acc = results["Text Only (Bio Masked)"]
    bio_acc  = results["Bio Only (Text Masked)"]

    print("\n" + "📊 " + "="*15 + " SUMMARY & CONCLUSION " + "="*15)
    print(f"1. Original Accuracy : {orig_acc:.4f}")
    print(f"2. Text Only Accuracy: {text_acc:.4f} (Drop: {orig_acc - text_acc:.4f})")
    print(f"3. Bio Only Accuracy : {bio_acc:.4f}  (Drop: {orig_acc - bio_acc:.4f})")
    
    diff = text_acc - bio_acc
    print("-" * 60)
    
    if diff > 0.1:
        print("🚨 [결론] 모델이 **텍스트(Text)**에 편향됨.")
    elif diff < -0.1:
        print("🚨 [결론] 모델이 **생체신호(Bio)**에 편향됨.")
    else:
        print("✅ [결론] 모델이 **균형** 있게 학습됨.")
    print("="*60 + "\n")

# ==========================================
# 🚀 메인 실행부
# ==========================================
def main():
    device = get_device()
    print(f"Running on device: {device}")

    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    print("Loading Data...")
    _, _, test_loader = load_multimodal_data(
        tokenizer=tokenizer,
        session_folder=config["paths"]["session_folder"],
        text_folder=config["paths"]["text_folder"],
        batch_size=config["training"]["batch_size"],
        max_len=config["model"]["max_len"]
    )

    model = MultimodalEndToEnd(config).to(device)

    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: {model_path} not found!")
        return

    evaluate_modality_bias(model, test_loader, device, tokenizer)

if __name__ == "__main__":
    main()