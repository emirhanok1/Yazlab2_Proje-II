import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Root dizini yola ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders import load_skab, load_batadal
from src.data.splits import skab_split, batadal_split
from src.data.preprocess import fit_scaler, apply_scaler, fit_pca, apply_pca, make_windows
from src.models.automata.paa_sax import PAASAXTransformer
from src.models.automata.automata import ProbabilisticAutomata

def get_best_threshold_f1(log_probs, y_true):
    """Validation seti uzerinde best F1 veren threshold ve YONU bulur."""
    best_f1 = -1
    best_thresh = None
    best_direction = None # 'low' veya 'high'
    percentiles = np.linspace(0, 100, 101) # 0'dan 100'e yuzdelikler
    
    for p in percentiles:
        thresh = np.percentile(log_probs, p)
        # Yon 1: Dusuk log-prob = anomali
        preds_low = (log_probs < thresh).astype(int)
        f1_low = f1_score(y_true, preds_low, zero_division=0)
        if f1_low > best_f1:
            best_f1 = f1_low
            best_thresh = thresh
            best_direction = 'low'
            
        # Yon 2: Yuksek log-prob = anomali
        preds_high = (log_probs > thresh).astype(int)
        f1_high = f1_score(y_true, preds_high, zero_division=0)
        if f1_high > best_f1:
            best_f1 = f1_high
            best_thresh = thresh
            best_direction = 'high'
            
    return best_thresh, best_direction, best_f1

def analyze_automata_dataset(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, config):
    print(f"\n{'='*50}\n{dataset_name.upper()} DATASET ANALIZI\n{'='*50}")
    
    # 1. Pipeline: Scaler, PCA, SAX
    scaler = fit_scaler(X_train, config)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    
    pca = fit_pca(X_train_s, config)
    pc1_train = apply_pca(X_train_s, pca)
    pc1_val = apply_pca(X_val_s, pca)
    pc1_test = apply_pca(X_test_s, pca)
    
    sax = PAASAXTransformer(config)
    train_symbols = sax.fit_transform(pc1_train)
    val_symbols = sax.transform(pc1_val)
    test_symbols = sax.transform(pc1_test)
    
    auto = ProbabilisticAutomata(config)
    auto.fit(train_symbols)
    
    # 2. Window (Y etiketleri) Hizalamasi
    _, y_val_w = make_windows(np.zeros((len(y_val), 1)), y_val, config)
    _, y_test_w = make_windows(np.zeros((len(y_test), 1)), y_test, config)
    
    # 3. Skorlama
    val_log_probs = auto._score_symbols(val_symbols)
    test_log_probs = auto._score_symbols(test_symbols)
    
    # Boyut hizalama (N - paa_segments + 1 - window_size + 1 = min_len)
    min_len_val = min(len(val_log_probs), len(y_val_w))
    y_val_a = y_val_w[-min_len_val:]
    val_probs_a = val_log_probs[-min_len_val:]
    
    min_len_test = min(len(test_log_probs), len(y_test_w))
    y_test_a = y_test_w[-min_len_test:]
    test_probs_a = test_log_probs[-min_len_test:]
    
    # 4. YON ANALIZI (Sadece test seti uzerinden bulguyu gormek icin)
    normal_p = test_probs_a[y_test_a == 0]
    anomaly_p = test_probs_a[y_test_a == 1]
    
    mean_normal = np.mean(normal_p) if len(normal_p)>0 else 0
    mean_anomaly = np.mean(anomaly_p) if len(anomaly_p)>0 else 0
    print(f"YON ANALIZI (Test Seti):")
    print(f"  Normal Pencereler Ortalama Log-Prob:  {mean_normal:.4f}")
    print(f"  Anomali Pencereler Ortalama Log-Prob: {mean_anomaly:.4f}")
    if mean_anomaly > mean_normal:
        print("  BULGU: Anomaliler DAHA YUKSEK olasilikli! (Nadir varsayimi GECERSIZ)")
    else:
        print("  BULGU: Anomaliler DAHA DUSUK olasilikli! (Nadir varsayimi GECERLI)")
        
    # 5. Validation'dan ESIK VE YON SECIMI
    best_thresh, best_direction, val_best_f1 = get_best_threshold_f1(val_probs_a, y_val_a)
    print(f"\nVALIDATION ESIK SECIMI:")
    print(f"  Secilen Threshold (Log-Prob): {best_thresh:.4f}")
    print(f"  Secilen Yon (Anomali Yönü):   {best_direction.upper()}")
    print(f"  Validation F1 (Max):          {val_best_f1:.4f}")
    
    # 6. Test Setine UYGULAMA (Automata icindeki kural ile)
    auto.set_threshold(best_thresh, best_direction)
    # auto.predict, window_size kadar elemani keserek labels uretir (last-step alignment)
    # test_symbols uzunlugu = N - paa_segments + 1. 
    # predict donusu = N - paa_segments + 1 - window_size + 1. (Bu da y_test_a ile ayni boydur)
    test_preds = auto.predict(test_symbols)
    
    t_acc = accuracy_score(y_test_a, test_preds)
    t_prec = precision_score(y_test_a, test_preds, zero_division=0)
    t_rec = recall_score(y_test_a, test_preds, zero_division=0)
    t_f1 = f1_score(y_test_a, test_preds, zero_division=0)
    
    print(f"\nTEST SONUCLARI (Secilen Threshold ile):")
    print(f"  Accuracy:  {t_acc:.4f}")
    print(f"  Precision: {t_prec:.4f}")
    print(f"  Recall:    {t_rec:.4f}")
    print(f"  F1 Score:  {t_f1:.4f}")

def main():
    set_seed(42)
    config = load_config()
    
    # --- SKAB ---
    X_s, y_s, grp_s = load_skab(config)
    skab_folds = list(skab_split(X_s, y_s, grp_s, config))
    
    # SKAB icin K-Fold'un birini (tr, te) alacagiz. 
    # Val icin train'in icinden (veya diger bir fold'u) secmeliyiz.
    # GroupKFold oldugu icin 0. fold train, 1. fold val, 2. fold test yapalim.
    tr_idx, _ = skab_folds[0]
    val_idx, _ = skab_folds[1] # aslinda tuple'in ikincisi test ama biz onu val gibi kullanalim
    te_idx = skab_folds[0][1]  # 0. foldun test kismi gercek test olsun.
    
    # Validation icin 1. fold'un "test" kismini kullaniyoruz (boylece gruplar ayrismis olur)
    val_idx = skab_folds[1][1]
    
    X_tr_s, y_tr_s = X_s.iloc[tr_idx].values, y_s[tr_idx]
    X_val_s, y_val_s = X_s.iloc[val_idx].values, y_s[val_idx]
    X_te_s, y_te_s = X_s.iloc[te_idx].values, y_s[te_idx]
    
    analyze_automata_dataset("SKAB", X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s, config)
    
    # --- BATADAL ---
    # Loader'i guvenli cagiralim:
    batadal_data = load_batadal(config)
    X_b, y_b = batadal_data[0], batadal_data[1]
    if len(batadal_data) == 3:
        dates_b = batadal_data[2]
    else:
        dates_b = batadal_data[3] if len(batadal_data) > 3 else None
        
    b_splits = batadal_split(X_b, y_b, dates_b, config)
    # Zaman sirali split: (train, val, test) donebilir. Phase 2'de batadal_split tr,val,te idx donuyordu.
    # Bakalim tuple mi list of tuple mi.
    try:
        tr_b, val_b, te_b = b_splits
    except:
        tr_b, val_b, te_b = b_splits[0]
        
    X_tr_b, y_tr_b = X_b.iloc[tr_b].values, y_b[tr_b]
    X_val_b, y_val_b = X_b.iloc[val_b].values, y_b[val_b]
    X_te_b, y_te_b = X_b.iloc[te_b].values, y_b[te_b]
    
    analyze_automata_dataset("BATADAL", X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b, config)

if __name__ == "__main__":
    main()
