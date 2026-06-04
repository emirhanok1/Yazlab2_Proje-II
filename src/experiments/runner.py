import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path

# Sklearn uyarilarini sustur
warnings.filterwarnings("ignore", category=UserWarning)

# Root dizinini yola ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import TensorDataset, DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders import load_skab, load_batadal
from src.data.splits import skab_split, batadal_split
from src.data.preprocess import (
    fit_scaler, apply_scaler, fit_pca, apply_pca, add_gaussian_noise, make_windows
)
from src.models.deep.lstm import LSTMModel
from src.models.deep.gru import GRUModel
from src.models.deep.cnn1d import CNN1DModel
from src.models.deep.trainer import DeepTrainer
from src.models.automata.paa_sax import PAASAXTransformer
from src.models.automata.automata import ProbabilisticAutomata

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, y_probs, model_name=""):
    """Metrikleri ve tahmin dagilimini hesaplar."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    unique, counts = np.unique(y_pred, return_counts=True)
    dist = {str(k): int(v) for k, v in zip(unique, counts)}
    is_degenerate = len(unique) <= 1
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "distribution": dist,
        "degenerate": is_degenerate
    }

def create_dl_model(model_name, n_features, config):
    if model_name == "lstm":
        return LSTMModel(config, n_features)
    elif model_name == "gru":
        return GRUModel(config, n_features)
    elif model_name == "cnn1d":
        return CNN1DModel(config, n_features)
    else:
        raise ValueError(f"Bilinmeyen DL model: {model_name}")

def run_deep_experiment(model_name, X_train, y_train, X_val, y_val, X_test, y_test, config, device):
    """Derin ogrenme modellerini calistirir (PCA kullanilmaz)."""
    scaler = fit_scaler(X_train, config)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    
    X_train_w, y_train_w = make_windows(X_train_s, y_train, config)
    X_val_w, y_val_w = make_windows(X_val_s, y_val, config)
    X_test_w, y_test_w = make_windows(X_test_s, y_test, config)
    
    n_features = X_train_w.shape[2]
    model = create_dl_model(model_name, n_features, config).to(device)
    
    pos_weight = None
    if config['models'].get('use_pos_weight', True):
        n_pos = np.sum(y_train_w == 1)
        n_neg = len(y_train_w) - n_pos
        if n_pos > 0:
            pos_weight = float(n_neg / n_pos)
            
    batch_size = config['training'].get('batch_size', 32)
    train_ds = TensorDataset(torch.tensor(X_train_w, dtype=torch.float32), torch.tensor(y_train_w, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_w, dtype=torch.float32), torch.tensor(y_val_w, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test_w, dtype=torch.float32), torch.tensor(y_test_w, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    trainer = DeepTrainer(model, config, train_loader, val_loader, pos_weight, device)
    trainer.train()
    
    # PREDICTION
    model.eval()
    all_preds = []
    all_probs = []
    t_inf_0 = time.time()
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device).float()
            logits = model(X_b)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            
    inf_time_ms = (time.time() - t_inf_0) * 1000.0
    
    metrics = compute_metrics(y_test_w, all_preds, all_probs, model_name.upper())
    metrics['train_time_ms'] = getattr(trainer, 'total_train_time_ms', 0)
    metrics['inf_time_ms'] = inf_time_ms
    
    return metrics

def get_best_threshold_f1(log_probs, y_true):
    best_f1 = -1
    best_thresh = None
    best_direction = None
    percentiles = np.linspace(0, 100, 101)
    
    for p in percentiles:
        thresh = np.percentile(log_probs, p)
        # Yon 1: Low
        preds_low = (log_probs < thresh).astype(int)
        f1_low = f1_score(y_true, preds_low, zero_division=0)
        if f1_low > best_f1:
            best_f1 = f1_low
            best_thresh = thresh
            best_direction = 'low'
            
        # Yon 2: High
        preds_high = (log_probs > thresh).astype(int)
        f1_high = f1_score(y_true, preds_high, zero_division=0)
        if f1_high > best_f1:
            best_f1 = f1_high
            best_thresh = thresh
            best_direction = 'high'
            
    return best_thresh, best_direction, best_f1

def run_automata_experiment(X_train, y_train, X_val, y_val, X_test, y_test, config):
    t0 = time.time()
    
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
    train_time = (time.time() - t0) * 1000.0
    
    # Validation
    val_log_probs = auto._score_symbols(val_symbols)
    _, y_val_w = make_windows(np.zeros((len(y_val), 1)), y_val, config)
    min_len_val = min(len(val_log_probs), len(y_val_w))
    y_val_a = y_val_w[-min_len_val:]
    val_probs_a = val_log_probs[-min_len_val:]
    
    best_thresh, best_direction, val_f1 = get_best_threshold_f1(val_probs_a, y_val_a)
    auto.set_threshold(best_thresh, best_direction)
    
    # Test
    t1 = time.time()
    preds, probs = auto.predict(test_symbols, return_scores=True)
    inf_time = (time.time() - t1) * 1000.0
    
    _, y_test_w = make_windows(np.zeros((len(y_test), 1)), y_test, config)
    min_len = min(len(preds), len(y_test_w))
    y_test_a = y_test_w[-min_len:]
    preds_a = preds[-min_len:]
    probs_a = probs[-min_len:]
    
    metrics = compute_metrics(y_test_a, preds_a, probs_a, "AUTOMATA")
    metrics['train_time_ms'] = train_time
    metrics['inf_time_ms'] = inf_time
    metrics['auto_direction'] = best_direction
    metrics['auto_threshold'] = best_thresh
    
    return metrics

def run_full_matrix(config, device):
    print("\n" + "="*50)
    print("  TAM DENEY MATRISI (FAZ 6)")
    print("="*50)
    
    # Checkpoint
    os.makedirs("results", exist_ok=True)
    ckpt_file = "results/experiments_checkpoint.json"
    results_list = []
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "r") as f:
            results_list = json.load(f)
        print(f">> Checkpoint bulundu, {len(results_list)} deney yendi atlanacak.")
    
    def is_done(m, d, sc, s, f):
        for r in results_list:
            if r['model']==m and r['dataset']==d and r['scenario']==sc and r['seed']==s and r['fold']==f:
                return True
        return False
        
    def save_ckpt():
        with open(ckpt_file, "w") as f:
            json.dump(results_list, f, indent=2)

    seeds = [42, 123, 2026, 7, 999]
    models = ["lstm", "gru", "cnn1d", "automata"]
    datasets = ["SKAB", "BATADAL"]
    scenarios = ["original", "gaussian_noise"] # unseen (cross dataset) is handled separately
    
    total_runs = len(seeds) * len(models) * 2 * len(scenarios) * 5 # Roughly
    print(f">> Toplam Beklenen Maksimum Deney: ~{total_runs}")
    
    for seed in seeds:
        set_seed(seed)
        
        # Datasets
        # SKAB
        X_skab, y_skab, groups = load_skab(config)
        skab_splits = list(skab_split(X_skab, y_skab, groups, config))
        
        # BATADAL
        batadal_data = load_batadal(config)
        X_bat, y_bat = batadal_data[0], batadal_data[1]
        dates_b = batadal_data[2] if len(batadal_data)==3 else None
        b_splits = batadal_split(X_bat, y_bat, dates_b, config)
        if isinstance(b_splits, tuple) and len(b_splits) == 3: # if returns tr,val,te
            batadal_splits = [(b_splits[0], b_splits[1], b_splits[2])]
        else:
            batadal_splits = [b_splits[0]] # Just first if list
            if len(batadal_splits[0]) == 2:
                # Mock validation
                tr_b, te_b = batadal_splits[0]
                batadal_splits = [(tr_b, te_b, te_b)]
        
        for ds_name, X_full, y_full, splits in [("SKAB", X_skab, y_skab, skab_splits), 
                                               ("BATADAL", X_bat, y_bat, batadal_splits)]:
            for fold_idx, split_tuple in enumerate(splits):
                if len(split_tuple) == 3:
                    tr_idx, val_idx, te_idx = split_tuple
                else:
                    tr_idx, te_idx = split_tuple
                    val_idx = te_idx # if no separate val
                    
                X_tr, y_tr = X_full.iloc[tr_idx].values, y_full[tr_idx]
                X_val, y_val = X_full.iloc[val_idx].values, y_full[val_idx]
                X_te, y_te = X_full.iloc[te_idx].values, y_full[te_idx]
                
                for scenario in scenarios:
                    X_tr_sc, X_val_sc, X_te_sc = X_tr, X_val, X_te
                    
                    if scenario == "gaussian_noise":
                        # Noise only added to test set typically to see robustness, 
                        # or added to all. Let's add to test set to test robustness.
                        X_te_sc = add_gaussian_noise(X_te, config)
                    
                    for model in models:
                        if is_done(model, ds_name, scenario, seed, fold_idx):
                            continue
                            
                        print(f"[{time.strftime('%H:%M:%S')}] Calisiyor: {model.upper()} | DS: {ds_name} | Senaryo: {scenario} | Seed: {seed} | Fold: {fold_idx}")
                        try:
                            if model == "automata":
                                res = run_automata_experiment(X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, config)
                            else:
                                res = run_deep_experiment(model, X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, config, device)
                                
                            run_info = {
                                "model": model,
                                "dataset": ds_name,
                                "scenario": scenario,
                                "seed": seed,
                                "fold": fold_idx,
                                **res
                            }
                            results_list.append(run_info)
                            save_ckpt()
                        except Exception as e:
                            print(f"HATA olustu ({model}, {ds_name}): {str(e)}")

    # Convert results to DataFrame and save
    df = pd.DataFrame(results_list)
    df.to_csv("results/experiments_final.csv", index=False)
    print("\n>>> Standart Deneyler Tamamlandi!")
    
    # === TABLO 4: Automata Sweep (Sadece SKAB, Original, Seed 42, Fold 0) ===
    print("\n" + "="*50)
    print("  AUTOMATA PARAMETRE TARAMASI (TABLO 4)")
    print("="*50)
    sweep_results = []
    set_seed(42)
    tr_idx, te_idx = skab_splits[0]
    val_idx = te_idx
    X_tr_sw, y_tr_sw = X_skab.iloc[tr_idx].values, y_skab[tr_idx]
    X_val_sw, y_val_sw = X_skab.iloc[val_idx].values, y_skab[val_idx]
    X_te_sw, y_te_sw = X_skab.iloc[te_idx].values, y_skab[te_idx]
    
    for w_size in [3, 4, 5, 6]:
        for a_size in [3, 4, 5, 6]:
            config_copy = json.loads(json.dumps(config))
            config_copy['automata']['window_size'] = w_size
            config_copy['automata']['alphabet_size'] = a_size
            
            print(f"Sweep: window={w_size}, alphabet={a_size}")
            try:
                res = run_automata_experiment(X_tr_sw, y_tr_sw, X_val_sw, y_val_sw, X_te_sw, y_te_sw, config_copy)
                res['window_size'] = w_size
                res['alphabet_size'] = a_size
                sweep_results.append(res)
            except Exception as e:
                print(f"Hata Sweep {w_size},{a_size}: {e}")
                
    pd.DataFrame(sweep_results).to_csv("results/automata_sweep.csv", index=False)
    
    # === TABLO 3: Cross-Dataset (Automata Sadece) ===
    print("\n" + "="*50)
    print("  CROSS-DATASET (TABLO 3)")
    print("="*50)
    cross_results = []
    
    def get_pc1(X_tr, X_val, X_te, cfg):
        sc = fit_scaler(X_tr, cfg)
        pca = fit_pca(apply_scaler(X_tr, sc), cfg)
        return apply_pca(apply_scaler(X_tr, sc), pca), \
               apply_pca(apply_scaler(X_val, sc), pca), \
               apply_pca(apply_scaler(X_te, sc), pca)
               
    tr_b, val_b, te_b = batadal_splits[0]
    X_tr_b, y_tr_b = X_bat.iloc[tr_b].values, y_bat[tr_b]
    X_val_b, y_val_b = X_bat.iloc[val_b].values, y_bat[val_b]
    X_te_b, y_te_b = X_bat.iloc[te_b].values, y_bat[te_b]

    # PC1 hesaplamalari kendi iclerinde kendi PCA'leri ile
    pc1_tr_s, pc1_val_s, pc1_te_s = get_pc1(X_tr_sw, X_val_sw, X_te_sw, config)
    pc1_tr_b, pc1_val_b, pc1_te_b = get_pc1(X_tr_b, X_val_b, X_te_b, config)
    
    def run_pc1_automata(pc1_tr, y_tr, pc1_val, y_val, pc1_te, y_te, cfg):
        t0 = time.time()
        sax = PAASAXTransformer(cfg)
        auto = ProbabilisticAutomata(cfg)
        train_symbols = sax.fit_transform(pc1_tr)
        auto.fit(train_symbols)
        val_symbols = sax.transform(pc1_val)
        val_probs = auto._score_symbols(val_symbols)
        _, y_val_w = make_windows(np.zeros((len(y_val), 1)), y_val, cfg)
        min_v = min(len(val_probs), len(y_val_w))
        best_t, best_d, _ = get_best_threshold_f1(val_probs[-min_v:], y_val_w[-min_v:])
        auto.set_threshold(best_t, best_d)
        t1 = time.time()
        preds, probs = auto.predict(sax.transform(pc1_te), return_scores=True)
        _, y_te_w = make_windows(np.zeros((len(y_te), 1)), y_te, cfg)
        min_t = min(len(preds), len(y_te_w))
        mets = compute_metrics(y_te_w[-min_t:], preds[-min_t:], probs[-min_t:], "CROSS")
        mets['train_time_ms'] = (t1 - t0)*1000
        mets['inf_time_ms'] = (time.time() - t1)*1000
        return mets
        
    try:
        print("SKAB (Egitim) -> BATADAL (Test)")
        res_s2b = run_pc1_automata(pc1_tr_s, y_tr_sw, pc1_val_s, y_val_sw, pc1_te_b, y_te_b, config)
        res_s2b['train_dataset'] = "SKAB"
        res_s2b['test_dataset'] = "BATADAL"
        cross_results.append(res_s2b)
    except Exception as e: print("Hata SKAB->BATADAL:", e)
        
    try:
        print("BATADAL (Egitim) -> SKAB (Test)")
        res_b2s = run_pc1_automata(pc1_tr_b, y_tr_b, pc1_val_b, y_val_b, pc1_te_s, y_te_sw, config)
        res_b2s['train_dataset'] = "BATADAL"
        res_b2s['test_dataset'] = "SKAB"
        cross_results.append(res_b2s)
    except Exception as e: print("Hata BATADAL->SKAB:", e)
        
    pd.DataFrame(cross_results).to_csv("results/cross_dataset.csv", index=False)

    print("\n>>> Tum Deneyler Tamamlandi! Sonuclar results/ altina kaydedildi.")
    
def main():
    parser = argparse.ArgumentParser(description="Faz 6 Deney Runner")
    parser.add_argument("--test", action="store_true", help="Kucuk test")
    parser.add_argument("--run", action="store_true", help="Tam matrisi baslat")
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.run:
        run_full_matrix(config, device)
    else:
        print("Parametre verilmedi. --run ile baslatin.")

if __name__ == "__main__":
    main()
