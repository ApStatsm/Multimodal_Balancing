import torch
from kobert_tokenizer import KoBERTTokenizer
from dataset import load_data_from_folders
from utils import get_device

def debug_data():
    # ==========================================
    # ❗ 본인의 경로로 수정해주세요
    # ==========================================
    csv_path = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data"
    text_folder = r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/KEMDy19_v1_3/wav"
    # ==========================================

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    print("🔍 데이터 로딩 중... (잠시만 기다려주세요)")
    train_loader, _ = load_data_from_folders(
        tokenizer=tokenizer,
        csv_path=csv_path,
        text_folder=text_folder,
        batch_size=16
    )

    print("\n" + "="*50)
    print("📢 [데이터 X-Ray 검사] 모델이 실제로 보는 텍스트")
    print("="*50)

    # 배치를 하나 뽑아서 내용물 확인
    for batch in train_loader:
        texts = batch['text']
        labels = batch['label']
        
        # 5개만 출력
        for i in range(5):
            print(f"\n[Sample {i+1}]")
            print(f"👉 Label (정답): {labels[i].item()}") # 0~4 숫자
            print(f"👉 Text  (입력): {texts[i]}")
        
        break  # 하나만 보고 종료

if __name__ == "__main__":
    debug_data()