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

def main():
    device = get_device()
    print(f"Running on Device: {device}")

    # 1. 데이터 로드 (DF 상태)
    train_df, test_df = load_data_frames(config["paths"]["session_folder"])
    
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    # 2. K-Fold CV 준비
    skf = StratifiedKFold(n_splits=config["training"]["k_folds"], shuffle=True, random_state=42)
    
    best_val_acc = 0.0
    best_model_state = None

    print(f"\n[Start {config['training']['k_folds']}-Fold Cross Validation]")

    # K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
        print(f"\n=== Fold {fold+1} ===")
        
        # Subset 생성
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        # DataLoader
        train_ds = MultimodalDataset(fold_train, config["paths"]["text_folder"], tokenizer)
        val_ds = MultimodalDataset(fold_val, config["paths"]["text_folder"], tokenizer)
        
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

        # 모델 초기화
        model = MultimodalEndToEnd(config).to(device)
        
        # Optimizer (전체 파라미터 학습 - LSTM Unfrozen)
        # KoBERT는 내부에서 frozen 처리됨
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config["training"]["learning_rate"]
        )

        # Training Loop
        for epoch in range(config["training"]["epochs"]):
            t_acc, t_loss = run_epoch(model, train_loader, optimizer, device, "train")
            v_acc, v_loss = run_epoch(model, val_loader, optimizer, device, "val")
            
            print(f"Ep {epoch+1:02d} | Train Acc: {t_acc:.3f} | Val Acc: {v_acc:.3f}")

        # Best Model 저장 (Validation 기준)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"⭐️ New Best Model (Val Acc: {best_val_acc:.3f})")

    # 3. Final Test (Best Model 사용)
    print("\n\n================ FINAL TEST REPORT ================")
    
    final_model = MultimodalEndToEnd(config).to(device)
    final_model.load_state_dict(best_model_state)
    
    test_ds = MultimodalDataset(test_df, config["paths"]["text_folder"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # (1) Normal Test
    acc_norm, _, _ = test_multimodal(final_model, test_loader, device, shuffle_mode="none")
    print(f"1. Normal Test Acc      : {acc_norm:.4f}")

    # (2) Text Shuffled (Bio Only)
    acc_shuf_txt, _, _ = test_multimodal(final_model, test_loader, device, shuffle_mode="text")
    print(f"2. Text Shuffled Acc    : {acc_shuf_txt:.4f} (Dependence on Bio)")

    # (3) Bio Shuffled (Text Only)
    acc_shuf_bio, _, _ = test_multimodal(final_model, test_loader, device, shuffle_mode="bio")
    print(f"3. Bio Shuffled Acc     : {acc_shuf_bio:.4f} (Dependence on Text)")
    
    # 결과 해석 힌트
    print("-" * 50)
    print("Tip: 점수가 많이 떨어질수록 모델이 해당 정보에 의존하고 있다는 뜻입니다.")

if __name__ == "__main__":
    main()