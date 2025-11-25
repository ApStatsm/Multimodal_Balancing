#kobert_classifier_project/
#│
#├── main.py
#├── dataset.py
#├── model.py
#├── train.py
#├── inference.py
#└── utils.py

# utils.py
import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# torch + torchvision (CUDA 환경에 맞춰 설치)
#pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
#pip install git+https://github.com/SKTBrain/KoBERT.git@master#egg=kobert_tokenizer&subdirectory=kobert_hf
# transformers / sentencepiece / kobert_tokenizer / gluonnlp
#pip install transformers==4.31.0
#pip install sentencepiece==0.1.99
#pip install kobert_tokenizer==0.1
#pip install gluonnlp==0.10.0

#  기타 필수 유틸
#pip install scikit-learn==1.3.0
#pip install pandas==2.0.3
#pip install tqdm==4.65.0
#pip install numpy==1.24.4
#pip install chardet
