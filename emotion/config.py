config = {
    "paths": {
        "session_folder": r'/Users/apstat/Desktop/02_연구/01_멀티모달 밸런싱/데이터',     #형준이가 준 sessiondata
        "text_folder": r'/Users/apstat/Desktop/02_연구/01_멀티모달 밸런싱/KEMDy20_v1_2/wav',           #wav 까지만 폴더 지정하면 하위 폴더 돌면서 txt 수집
    },

    "training": {
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 1e-4,
    },

    "model": {
        "max_len": 128,
        "bio_input_dim": 4,
        "bio_hidden_dim": 64,
        "bio_output_dim": 32,
        "fusion_hidden_dim": 256,
        "num_classes": 2
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
