from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import numpy as np

def skab_split(X, y, groups, config):
    """
    SKAB için cross-validation bölücüsü oluşturur.
    Öncelikle StratifiedGroupKFold dener, sınıf dengesizliğinde işe yarar.
    StratifiedGroupKFold çalışmazsa veya kullanıcı groupkfold istemişse GroupKFold'a düşer.
    Generator (yield) olarak (train_idx, test_idx) döndürür.
    """
    n_folds = config['split']['skab_n_folds']
    # stratify istendiğinde
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_folds)
        # Sadece test etmek için ilk fold alınabiliyor mu?
        _ = next(sgkf.split(X, y, groups))
        print(">> [INFO] StratifiedGroupKFold başarıyla kullanılıyor (SKAB).")
        return sgkf.split(X, y, groups)
    except Exception as e:
        print(f">> [WARN] StratifiedGroupKFold başarısız oldu: {e}. GroupKFold'a düşülüyor.")
        gkf = GroupKFold(n_splits=n_folds)
        return gkf.split(X, y, groups)

def batadal_split(X, y, time_series, config):
    """
    BATADAL veri setini kronolojik (zaman sıralı) olarak Train/Val/Test (%60, %20, %20) olarak böler.
    Rastgele karıştırma (shuffle) YAPILMAZ.
    Geriye index listeleri veya doğrudan veri parçaları döndürebiliriz;
    Burada (train_idx, val_idx, test_idx) listesi dönüyoruz.
    """
    n = len(X)
    train_ratio = config['split']['batadal']['train']
    val_ratio = config['split']['batadal']['val']
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    indices = np.arange(n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx
