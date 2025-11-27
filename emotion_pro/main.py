from config import config
from utils import get_device
from dataset_multimodal import load_data_frames, MultimodalDataset
from models.multimodal_e2e import MultimodalEndToEnd
# 🔥 train.py에서 run_epoch, test_multimodal, WeightedCrossEntropyLoss 임포트
from train import run_epoch, test_multimodal, WeightedCrossEntropyLoss 
import os
import torch
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer

# 📊 시각화 및 평가용 라이브러리
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# =============================================================================
# 🔥 가중치 계산 함수 (STAGE 2)
# =============================================================================
def calculate_weights_bio_only(preds_A, preds_B, labels):
    """
    Bio-Only 모델(B)의 오분류 집합(M)에 T 기반의 적응적 가중치를 부여합니다.
    Args:
        preds_A (np.array): 멀티모달 모델 A의 예측값 (Train Set 기준)
        preds_B (np.array): Bio-Only 모델 B의 예측값 (Train Set 기준)
        labels (np.array): 실제 레이블
    Returns:
        list: 샘플별 가중치 리스트
    """
    # 1. F1-score 계산
    f1_A = f1_score(labels, preds_A, average='macro', zero_division=0)
    f1_B = f1_score(labels, preds_B, average='macro', zero_division=0)
    
    # 2. 멀티모달 이득 T 계산 (Bias Measurement)
    T = f1_A - f1_B
    
    # 3. 적응적 가중치 계수 W_Adaptive 산출
    T_min = 0.001       # T의 최소 안정화 값 (0으로 나누는 것을 방지)
    epsilon = 1e-6      # 안정화 상수
    gamma = 1.0         # 가중치 튜닝 파라미터 (조절 가능)

    T_Stabilized = max(T, T_min) + epsilon
    
    # T가 작을수록 W_Adaptive는 커진다. (Bio-Only 성능과 A 성능 차이가 작을수록 Text 기여를 강제)
    W_Adaptive = gamma * np.log(1 + 1 / T_Stabilized)
    
    # 4. M 집합 (Bio-Only 모델이 틀린 샘플) 식별
    # B 모델이 틀렸을 때의 인덱스
    M_indices = np.where(preds_B != labels)[0] 
    
    # 5. 가중치 할당
    weights = np.ones(len(labels), dtype=np.float32)
    weights[M_indices] = 1.0 + W_Adaptive
    
    print(f"  F1_A (Multimodal): {f1_A:.4f}, F1_B (Bio-Only): {f1_B:.4f}")
    print(f"  T (Multimodal Gain): {T:.4f}, W_Adaptive Max Coef: {W_Adaptive:.4f}")
    print(f"  M Set Size: {len(M_indices)} / {len(labels)} ({len(M_indices)/len(labels)*100:.2f}%)")
    print(f"  Max Sample Weight: {1.0 + W_Adaptive:.4f}\n")
    
    return weights.tolist() 


def save_plots_and_report(scenario_name, labels, preds, probs):
    """
    결과 출력 및 시각화 저장 함수 (emotion/result 폴더에 저장)
    """
    output_dir = os.path.join("emotion", "result")
    os.makedirs(output_dir, exist_ok=True)  
    
    # 1. Classification Report
    target_names = ["Neutral", "Surprise", "Angry", "Sad", "Happy"]
    print(f"\n>> Classification Report ({scenario_name}):")
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    # 2. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {scenario_name}')
    plt.tight_layout()
    save_path_cm = os.path.join(output_dir, f'{scenario_name}_confusion_matrix.png')
    plt.savefig(save_path_cm)
    plt.close()
    print(f"   [Save] Confusion Matrix: {save_path_cm}")


def main():
    # 0. 설정 로드 및 장치/토크나이저 초기화
    device = get_device()
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # 1. 데이터 로드 (멀티모달 데이터 로드)
    train_df, test_df = load_data_frames(config["paths"]["session_folder"])
    
    # ================= 🚀 STAGE 1: 초기 멀티모달 모델 A 학습 (No Weights) =================
    print("\n\n=============== 🚀 STAGE 1: Initial Multimodal Training (Model A) ===============")
    
    # 1차 학습용 데이터셋 (가중치=None, 모두 1)
    train_ds_stage1 = MultimodalDataset(train_df, config["paths"]["text_folder"], tokenizer)
    train_loader = DataLoader(train_ds_stage1, batch_size=config["training"]["batch_size"], shuffle=True)
    
    # Test 로더 (가중치 미사용)
    test_ds = MultimodalDataset(test_df, config["paths"]["text_folder"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    print(f"Data Split: Train/Test Samples: {len(train_df)}/{len(test_df)}")

    # 모델 및 옵티마이저 초기화 (Model A)
    model_A = MultimodalEndToEnd(config).to(device)
    optimizer_A = torch.optim.AdamW(model_A.parameters(), lr=config["training"]["learning_rate"])
    
    for epoch in range(config["training"]["epochs"]):
        t_loss, t_acc = run_epoch(model_A, train_loader, optimizer_A, device, "train")
        v_acc, v_loss, _, _, _ = test_multimodal(model_A, test_loader, device, shuffle_mode="none")
        print(f"Ep {epoch+1:02d} (A) | Train Acc: {t_acc:.3f}, Loss: {t_loss:.4f} | Test Acc: {v_acc:.3f}")
        
    
    # ================= ⚖️ STAGE 2: 가중치 계산 및 M 식별 (Bio-Only Baseline) =================
    print("\n\n=============== ⚖️ STAGE 2: Adaptive Weight Calculation (Bio-Only Baseline) ===============")

    # 1. 가중치 계산을 위해 셔플링되지 않은 Train Loader 사용
    train_loader_unshuffled = DataLoader(train_ds_stage1, 
                                         batch_size=config["training"]["batch_size"], 
                                         shuffle=False)
    
    # 2. 멀티모달 F1_A 계산 (shuffle_mode="none")
    print("  Testing Model A (Multimodal) on Train Set...")
    _, _, labels_A, preds_A, _ = test_multimodal(model_A, train_loader_unshuffled, device, shuffle_mode="none")

    # 3. Bio-Only F1_B 계산 (shuffle_mode="text_zeroout")
    print("  Testing Model A (Bio-Only) on Train Set...")
    # 🔥 텍스트 인풋을 0으로 마스킹하여 Bio 모달리티만 사용하는 모델 B의 성능 시뮬레이션
    _, _, labels_B, preds_B, _ = test_multimodal(model_A, train_loader_unshuffled, device, shuffle_mode="text_zeroout") 

    # 4. 가중치 w_i 계산 및 추출
    new_weights = calculate_weights_bio_only(np.array(preds_A), np.array(preds_B), np.array(labels_A))


    # ================= 🔄 STAGE 3: 가중치 적용 모델 A' 재학습 =================
    print("\n\n=============== 🔄 STAGE 3: Weighted Re-Training (Model A') ===============")

    # 모델 파라미터 초기화 (새로운 모델로 시작하거나, A 모델을 복사하여 시작)
    model_A_prime = MultimodalEndToEnd(config).to(device)
    # A 모델의 가중치를 가져와서 시작
    model_A_prime.load_state_dict(model_A.state_dict()) 
    optimizer_A_prime = torch.optim.AdamW(model_A_prime.parameters(), lr=config["training"]["learning_rate"])
    
    # 1. 가중치가 적용된 새로운 데이터셋/로더 생성
    train_ds_stage2 = MultimodalDataset(train_df, config["paths"]["text_folder"], tokenizer, weights=new_weights)
    train_loader_stage2 = DataLoader(train_ds_stage2, batch_size=config["training"]["batch_size"], shuffle=True)
    
    # 2. 2차 학습 루프 (Weighted Training)
    for epoch in range(config["training"]["epochs"]):
        t_loss, t_acc = run_epoch(model_A_prime, train_loader_stage2, optimizer_A_prime, device, "train")
        v_acc, v_loss, _, _, _ = test_multimodal(model_A_prime, test_loader, device, shuffle_mode="none")
        print(f"Ep {epoch+1:02d} (A') | W-Train Acc: {t_acc:.3f}, Loss: {t_loss:.4f} | Test Acc: {v_acc:.3f}")


    # ================= FINAL TEST (Model A') =================
    print("\n\n================ FINAL EVALUATION (Model A') ================")
    
    # 최종 모델 A'을 사용
    final_model = model_A_prime 

    # 3가지 시나리오 정의 (주석을 해제하여 사용)
    scenarios = [
        ("1_Normal_Multimodal_Test", "none"),          # 정상 멀티모달 성능
        ("2_Bio_Only_(Text_Masked)", "text_zeroout"), # 텍스트 마스킹 (Bio-Only 성능 확인)
        ("3_Text_Only_(Bio_Masked)", "bio_zeroout")   # Bio 마스킹 (Text-Only 성능 확인)
    ]

    for name, mode in scenarios:
        print(f"\n🔶 Running Scenario: {name}")
        
        # 1. 테스트 실행
        acc, loss, labels, preds, probs = test_multimodal(final_model, test_loader, device, shuffle_mode=mode)
        
        print(f"   Test Acc: {acc:.4f}, Loss: {loss:.4f}")
        
        # 2. 결과 출력 및 시각화 저장 (필요 시 주석 해제)
        save_plots_and_report(name, labels, preds, probs) 
        
    print("\n✅ All stages completed.")


if __name__ == '__main__':
    main()