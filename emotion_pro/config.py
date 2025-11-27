config = {
    "paths": {
        "session_folder": r'/Users/apstat/Desktop/02_연구/Multimodal_Balancing/19data',
        "text_folder": r"/Users/apstat/Desktop/02_연구/Multimodal_Balancing/KEMDy19_v1_3/wav",
    },

    "training": {
        "batch_size": 32,
        "epochs": 10,        # 10 epoch
        "learning_rate": 3e-4
    },

    "model": {
        "max_len": 128,
        "bio_input_dim": 4,
        "bio_hidden_dim": 64,
        "bio_output_dim": 64,
        "fusion_hidden_dim": 256,
        "num_classes": 5     # 이진 분류 (Neutral vs Biased)
    }
}

#pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

#pip install transformers==4.31.0
#pip install sentencepiece==0.1.99

#pip install git+https://github.com/SKTBrain/KoBERT.git@master#egg=kobert_tokenizer&subdirectory=kobert_hf

#pip install gluonnlp==0.10.0
#pip install numpy==1.24.4
#pip install pandas==2.0.3
#pip install chardet==5.2.0
#pip install tqdm==4.65.0

#pip install scikit-learn==1.3.0
#pip install matplotlib==3.7.1
#pip install seaborn==0.12.2
