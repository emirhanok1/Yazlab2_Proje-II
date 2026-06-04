"""
test_automata_report.py — Otomata modeli uctan uca test ve rapor scripti.

Calistirma:
    python tests/test_automata_report.py

NOT: Bu dosya pytest tarafindan COLLECT edilmemeli (test fonksiyonlari yok).
     Tum kod if __name__ == "__main__" korumasi altindadir.

Raporlar:
  1. SAX dizisi ornegi
  2. State sayisi
  3. Gecis matrisi sekli
  4. Add-k oncesi/sonrasi sifir kontrolu
  5. Bir pencere icin log + normalize path-prob + guven skoru
  6. Train'den hesaplanan esik degeri
  7. Ornek pencere kararlari
  8. fit/transform leakage testi
"""

import sys
import os

# Proje kok dizinini path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import yaml
from collections import defaultdict

from src.models.automata.paa_sax import PAASAXTransformer
from src.models.automata.automata import (
    ProbabilisticAutomata,
    _symbols_to_patterns,
    _build_transition_counts,
    _build_transition_matrix,
)


def main():
    # ============================================================
    # 0. Config yukle
    # ============================================================
    config_path = os.path.join(ROOT, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    automata_cfg = config["automata"]
    print("=" * 60)
    print("CONFIG (automata bolumu):")
    for k, v in automata_cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ============================================================
    # 1. Sentetik PC1 sinyalleri uret (deterministik, seed sabit)
    # ============================================================
    rng = np.random.default_rng(42)

    N_TRAIN = 200
    N_TEST = 80

    train_signal = np.sin(np.linspace(0, 10 * np.pi, N_TRAIN)) + rng.normal(0, 0.1, N_TRAIN)
    test_signal = np.sin(np.linspace(0, 4 * np.pi, N_TEST)) + rng.normal(0, 0.1, N_TEST)
    anomaly_start = int(N_TEST * 0.8)
    test_signal[anomaly_start:] += rng.uniform(3, 5, N_TEST - anomaly_start)

    print(f"\nSinyal sekilleri -> Train: {train_signal.shape}, Test: {test_signal.shape}")

    # ============================================================
    # 2. PAA + SAX -- fit YALNIZ train'de
    # ============================================================
    paa_sax = PAASAXTransformer(config)
    print(f"\n{paa_sax}")

    train_symbols = paa_sax.fit_transform(train_signal)
    print(f"\n[1] SAX dizisi (train, ilk 20 sembol): {train_symbols[:20]}")
    print(f"    Toplam sembol sayisi (train): {len(train_symbols)}")
    print(f"    Alfabe: {sorted(set(train_symbols))}")

    test_symbols = paa_sax.transform(test_signal)
    print(f"\n[1] SAX dizisi (test, ilk 20 sembol): {test_symbols[:20]}")
    print(f"    Toplam sembol sayisi (test): {len(test_symbols)}")

    bp = paa_sax.get_breakpoints()
    print(f"\n    Breakpoint'ler (train'den ogrenildi, donmus): {bp}")

    # ============================================================
    # 3. FIT/TRANSFORM LEAKAGE TESTI
    # ============================================================
    print("\n" + "=" * 60)
    print("[8] FIT/TRANSFORM LEAKAGE TESTI")
    print("=" * 60)

    bp_before = paa_sax.get_breakpoints().copy()
    _ = paa_sax.transform(test_signal)
    bp_after = paa_sax.get_breakpoints()

    breakpoints_unchanged = np.allclose(bp_before, bp_after)
    print(f"  transform(test) sonrasi breakpoint'ler degismedi: {breakpoints_unchanged}")
    assert breakpoints_unchanged, "HATA: Test transform'u breakpoint'leri degistirdi!"

    paa_sax_leakage_test = PAASAXTransformer(config)
    paa_sax_leakage_test.fit(train_signal)
    bp_train_only = paa_sax_leakage_test.get_breakpoints().copy()

    paa_sax_leakage_test2 = PAASAXTransformer(config)
    paa_sax_leakage_test2.fit(test_signal)
    bp_test_only = paa_sax_leakage_test2.get_breakpoints().copy()

    leakage_would_change = not np.allclose(bp_train_only, bp_test_only)
    print(f"  Test uzerinde fit() olsaydi breakpoint'ler farkli miydi: {leakage_would_change}")
    print(f"    Train BP: {np.round(bp_train_only, 4)}")
    print(f"    Test  BP: {np.round(bp_test_only, 4)}")
    print(f"  [OK] Dogru kullanim: yalnizca transform(test) -> breakpoint'ler KORUNUYOR")

    # ============================================================
    # 4. Otomata fit -- YALNIZ train SAX dizisi
    # ============================================================
    print("\n" + "=" * 60)
    print("[2] OTOMATA FIT (yalniz train)")
    print("=" * 60)

    model = ProbabilisticAutomata(config)
    model.fit(train_symbols)

    print(f"  State sayisi: {model.n_states}")
    print(f"  Gecis matrisi sekli: {model.transition_matrix_shape}")
    print(f"  Model: {model}")

    # ============================================================
    # 5. Add-k oncesi / sonrasi sifir kontrolu
    # ============================================================
    print("\n" + "=" * 60)
    print("[4] ADD-K SMOOTHING SIFIR KONTROLU")
    print("=" * 60)

    train_patterns = _symbols_to_patterns(train_symbols, model.window_size)
    state_idx, raw_counts = _build_transition_counts(train_patterns)

    M = len(state_idx)
    count_matrix = np.zeros((M, M), dtype=float)
    for (src, dst), cnt in raw_counts.items():
        i = state_idx[src]
        j = state_idx[dst]
        count_matrix[i, j] = cnt

    zeros_before = int(np.sum(count_matrix == 0))
    zeros_after = int(np.sum(model.trans_matrix_ == 0))

    print(f"  Toplam olasi gecis cifti: {M} x {M} = {M*M}")
    print(f"  Gozlemlenen gecis sayisi: {len(raw_counts)}")
    print(f"  Smoothing ONCE sifir sayisi: {zeros_before}")
    print(f"  Smoothing SONRA sifir sayisi: {zeros_after}")
    assert zeros_after == 0, "HATA: Add-k sonrasi hala sifir var!"
    print(f"  [OK] Add-k smoothing sonrasi matris tamamen sifirsiz")

    # ============================================================
    # 6. Bir pencere icin log + normalize path-prob + guven skoru
    # ============================================================
    print("\n" + "=" * 60)
    print("[5] BIR PENCERE ORNEGI: LOG PATH-PROB + GUVEN SKORU")
    print("=" * 60)

    example_window_symbols = train_symbols[:automata_cfg["paa_segments"]]
    print(f"  Ornek pencere sembolleri: {example_window_symbols}")

    log_p = model.window_log_prob(example_window_symbols, normalize=True)
    conf = model.confidence_score(log_p)

    print(f"  Normalize log-prob (per-transition avg): {log_p:.6f}")
    print(f"  Guven skoru [0,1]: {conf:.4f}  (yuksek = normal)")
    print(f"  Karar: {'NORMAL' if log_p >= model.threshold_ else 'ANOMALI'}")

    # ============================================================
    # 7. Train'den hesaplanan esik degeri
    # ============================================================
    print("\n" + "=" * 60)
    print("[6] TRAIN ESIGI")
    print("=" * 60)

    print(f"  anomaly_percentile: {model.anomaly_percentile}")
    print(f"  Train normalize log-prob -> min: {model.train_scores_.min():.6f}")
    print(f"  Train normalize log-prob -> mean: {model.train_scores_.mean():.6f}")
    print(f"  Train normalize log-prob -> max: {model.train_scores_.max():.6f}")
    print(f"  Esik ({model.anomaly_percentile}. percentile): {model.threshold_:.6f}")

    # ============================================================
    # 8. Ornek pencere kararlari (test verisi)
    # ============================================================
    print("\n" + "=" * 60)
    print("[7] ORNEK PENCERE KARARLARI (TEST)")
    print("=" * 60)

    labels, raw_scores = model.predict(test_symbols, return_scores=True)
    proba = model.predict_proba(test_symbols)

    n_show = min(10, len(labels))
    print(f"  {'Index':<8} {'NormLogProb':>14} {'AnomalProba':>13} {'Karar':>10}")
    print(f"  {'-'*8} {'-'*14} {'-'*13} {'-'*10}")
    for idx in range(n_show):
        karar = "ANOMALI" if labels[idx] == 1 else "normal"
        print(f"  {idx:<8} {raw_scores[idx]:>14.6f} {proba[idx]:>13.4f} {karar:>10}")

    print(f"\n  ... (son {n_show} pencere -- anomali bolgesi)")
    print(f"  {'Index':<8} {'NormLogProb':>14} {'AnomalProba':>13} {'Karar':>10}")
    print(f"  {'-'*8} {'-'*14} {'-'*13} {'-'*10}")
    for idx in range(max(0, len(labels) - n_show), len(labels)):
        karar = "ANOMALI" if labels[idx] == 1 else "normal"
        print(f"  {idx:<8} {raw_scores[idx]:>14.6f} {proba[idx]:>13.4f} {karar:>10}")

    n_anomaly = int(labels.sum())
    print(f"\n  Toplam pencere: {len(labels)}, Tespit edilen anomali: {n_anomaly}")

    # ============================================================
    # OZET
    # ============================================================
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"  SAX alfabe buyuklugu       : {automata_cfg['alphabet_size']}")
    print(f"  PAA segment sayisi         : {automata_cfg['paa_segments']}")
    print(f"  Sliding window_size        : {automata_cfg['window_size']}")
    print(f"  Smoothing_k                : {automata_cfg['smoothing_k']}")
    print(f"  Anomaly percentile         : {automata_cfg['anomaly_percentile']}")
    print(f"  Train SAX sembolleri       : {len(train_symbols)}")
    print(f"  Test  SAX sembolleri       : {len(test_symbols)}")
    print(f"  State (unique pattern) sayisi: {model.n_states}")
    print(f"  Gecis matrisi sekli        : {model.transition_matrix_shape}")
    print(f"  Smoothing oncesi sifir     : {zeros_before}")
    print(f"  Smoothing sonrasi sifir    : {zeros_after}")
    print(f"  Esik degeri                : {model.threshold_:.6f}")
    print(f"  Leakage testi              : GECTI")
    print("=" * 60)
    print("TUM TESTLER BASARILI")


if __name__ == "__main__":
    main()
