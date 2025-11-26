import yaml
import torch

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device():
    """
    우선순위: CUDA (NVIDIA) > MPS (Mac M1~M4) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 맥북 실리콘(M1, M2, M3, M4) 지원 확인
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")