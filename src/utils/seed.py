import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Tekrarlanabilirlik (reproducibility) için rastgelelik tohumlarını (seed) sabitler.
    Python, NumPy ve PyTorch bileşenlerini kapsar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA ortamında deterministic sonuçlar için:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Ortam değişkeni (hash seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Seed {seed} olarak sabitlendi.")
