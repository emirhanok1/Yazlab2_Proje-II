import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    Belirtilen YAML dosyasını yükler ve bir sözlük (dictionary) olarak döndürür.
    Varsayılan olarak 'config/config.yaml' yolunu kullanır.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
