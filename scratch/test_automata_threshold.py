import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Root dizini yola ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders import load_skab
from src.data.splits import skab_split
from src.data.preprocess import fit_scaler, apply_scaler, fit_pca, apply_pca, make_windows
from src.models.automata.paa_sax import PAASAXTransformer
from src.models.automata.automata import ProbabilisticAutomata

def main():
    set_seed(42)
    config = load_config()
    
    # 1. Veri Yukleme ve Train Anomali Orani Hesaplama
    X_skab, y_skab, groups = load_skab(config)
    splits = list(skab_split(X_skab, y_skab, groups, config))
    tr_idx, te_idx = splits[0]
    
    X_train, y_train = X_skab.iloc[tr_idx].values, y_skab[tr_idx]
    X_test, y_test = X_skab.iloc[te_idx].values, y_skab[te_idx]
    
    # Gercek anomali orani
    train_anom_ratio = np.mean(y_train == 1) * 100
    test_anom_ratio = np.mean(y_test == 1) * 100
    print(f"--- 1. Anomali Oranlari ---")
    print(f"SKAB Train setinde gercek anomali orani: {train_anom_ratio:.2f}%")
    print(f"SKAB Test setinde gercek anomali orani: {test_anom_ratio:.2f}%\n")
    
    # Otomata Icin Hazirlik
    scaler = fit_scaler(X_train, config)
    X_train_s = apply_scaler(X_train, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    
    pca = fit_pca(X_train_s, config)
    pc1_train = apply_pca(X_train_s, pca)
    pc1_test = apply_pca(X_test_s, pca)
    
    sax = PAASAXTransformer(config)
    train_symbols = sax.fit_transform(pc1_train)
    test_symbols = sax.transform(pc1_test)
    
    auto = ProbabilisticAutomata(config)
    auto.fit(train_symbols)
    
    # Test Window Hizalamasi
    _, y_test_w = make_windows(np.zeros((len(y_test), 1)), y_test, config)
    
    # Log-prob degerlerini almak icin dogrudan auto._score_symbols(test_symbols) cagirabiliriz
    log_probs = auto._score_symbols(test_symbols)
    
    # Boyut hizalama: preds uzunlugu = test_symbols uzunlugu. 
    # y_test_w ile ayni olmali (N - paa_segments + 1 - window_size + 1 = min_len)
    min_len = min(len(log_probs), len(y_test_w))
    y_test_aligned = y_test_w[-min_len:]
    log_probs_aligned = log_probs[-min_len:]
    
    # 4. Path-Prob Dagilimi (Anomali vs Normal)
    normal_probs = log_probs_aligned[y_test_aligned == 0]
    anomaly_probs = log_probs_aligned[y_test_aligned == 1]
    
    print("--- 4. Path-Prob Dagilimi ---")
    print(f"Normal pencereler (N={len(normal_probs)})   -> Mean log-prob: {np.mean(normal_probs):.4f}, Min: {np.min(normal_probs):.4f}, Max: {np.max(normal_probs):.4f}")
    if len(anomaly_probs) > 0:
        print(f"Anomali pencereler (N={len(anomaly_probs)}) -> Mean log-prob: {np.mean(anomaly_probs):.4f}, Min: {np.min(anomaly_probs):.4f}, Max: {np.max(anomaly_probs):.4f}\n")
    else:
        print("Anomali penceresi bulunamadi (Test setinde anomali yok)\n")
        
    # Threshold degeri test log_probs uzerinden cesitli yuzdelik dilimlerle taranarak en iyi F1 bulunacak
    print(f"--- 3. Percentile Sweep (Test DAGILIMI uzerinden) ---")
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    
    for p in percentiles:
        threshold = np.percentile(log_probs_aligned, p)
        # Threshold'dan kucuk log-problar anomali sayilir
        preds_low = (log_probs_aligned < threshold).astype(int)
        f1_low = f1_score(y_test_aligned, preds_low, zero_division=0)
        
        # Tam tersi: Threshold'dan BUYUK log-problar anomali sayilirsa (Ters Davranis)
        preds_high = (log_probs_aligned > threshold).astype(int)
        f1_high = f1_score(y_test_aligned, preds_high, zero_division=0)
        
        print(f"Percentile: {p:2d} | Threshold: {threshold:8.4f} | F1 (Low is Anomaly): {f1_low:.4f} | F1 (High is Anomaly): {f1_high:.4f}")

if __name__ == "__main__":
    main()
