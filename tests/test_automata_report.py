"""
test_automata_report.py — Otomata modeli uçtan uca test ve rapor scripti.

Çalıştır:
    python tests/test_automata_report.py

Raporlar:
  1. SAX dizisi örneği
  2. State sayısı
  3. Geçiş matrisi şekli
  4. Add-k öncesi/sonrası sıfır kontrolü
  5. Bir pencere için log + normalize path-prob + güven skoru
  6. Train'den hesaplanan eşik değeri
  7. Örnek pencere kararları
  8. fit/transform leakage testi (test verisi breakpoint'leri değiştirmemeli)
"""

import sys
import os

# Proje kök dizinini path'e ekle
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

# ============================================================
# 0. Config yükle
# ============================================================
config_path = os.path.join(ROOT, "config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

automata_cfg = config["automata"]
print("=" * 60)
print("CONFIG (automata bölümü):")
for k, v in automata_cfg.items():
    print(f"  {k}: {v}")
print("=" * 60)

# ============================================================
# 1. Sentetik PC1 sinyalleri üret (deterministik, seed sabit)
# ============================================================
rng = np.random.default_rng(42)

N_TRAIN = 200
N_TEST = 80

# Train: normal periyodik sinyal
train_signal = np.sin(np.linspace(0, 10 * np.pi, N_TRAIN)) + rng.normal(0, 0.1, N_TRAIN)

# Test: normal + anomali bölgesi (son %20)
test_signal = np.sin(np.linspace(0, 4 * np.pi, N_TEST)) + rng.normal(0, 0.1, N_TEST)
anomaly_start = int(N_TEST * 0.8)
test_signal[anomaly_start:] += rng.uniform(3, 5, N_TEST - anomaly_start)  # sert sıçrama

print(f"\nSinyal şekilleri → Train: {train_signal.shape}, Test: {test_signal.shape}")

# ============================================================
# 2. PAA + SAX — fit YALNIZ train'de
# ============================================================
paa_sax = PAASAXTransformer(config)
print(f"\n{paa_sax}")

train_symbols = paa_sax.fit_transform(train_signal)
print(f"\n[1] SAX dizisi (train, ilk 20 sembol): {train_symbols[:20]}")
print(f"    Toplam sembol sayısı (train): {len(train_symbols)}")
print(f"    Alfabe: {sorted(set(train_symbols))}")

# Test'e transform (donmuş breakpoint)
test_symbols = paa_sax.transform(test_signal)
print(f"\n[1] SAX dizisi (test, ilk 20 sembol): {test_symbols[:20]}")
print(f"    Toplam sembol sayısı (test): {len(test_symbols)}")

# Breakpoint kontrolü
bp = paa_sax.get_breakpoints()
print(f"\n    Breakpoint'ler (train'den öğrenildi, donmuş): {bp}")

# ============================================================
# 3. FIT/TRANSFORM LEAKAGE TESTİ
# ============================================================
print("\n" + "=" * 60)
print("[8] FIT/TRANSFORM LEAKAGE TESTİ")
print("=" * 60)

# Aynı breakpoint'leri tekrar elde etmek için sadece train'i kullanıyoruz
bp_before = paa_sax.get_breakpoints().copy()

# Test verisini transform et → breakpoint'ler değişmemeli
_ = paa_sax.transform(test_signal)
bp_after = paa_sax.get_breakpoints()

breakpoints_unchanged = np.allclose(bp_before, bp_after)
print(f"  transform(test) sonrası breakpoint'ler değişmedi: {breakpoints_unchanged}")
assert breakpoints_unchanged, "HATA: Test transform'u breakpoint'leri değiştirdi! (Data leakage!)"

# fit() ile test verisi üzerinde tekrar fit edilmeye çalışılırsa breakpoint'ler değişir —
# bunu göster (ama gerçek kullanımda bunu YAPMA)
paa_sax_leakage_test = PAASAXTransformer(config)
paa_sax_leakage_test.fit(train_signal)
bp_train_only = paa_sax_leakage_test.get_breakpoints().copy()

paa_sax_leakage_test2 = PAASAXTransformer(config)
paa_sax_leakage_test2.fit(test_signal)  # YANLIŞ: sadece demo için
bp_test_only = paa_sax_leakage_test2.get_breakpoints().copy()

leakage_would_change = not np.allclose(bp_train_only, bp_test_only)
print(f"  Test üzerinde fit() olsaydı breakpoint'ler farklı mıydı: {leakage_would_change}")
print(f"    Train BP: {np.round(bp_train_only, 4)}")
print(f"    Test  BP: {np.round(bp_test_only, 4)}")
print(f"  ✓ Doğru kullanım: yalnızca transform(test) → breakpoint'ler KORUNUYOR")

# ============================================================
# 4. Otomata fit — YALNIZ train SAX dizisi
# ============================================================
print("\n" + "=" * 60)
print("[2] OTOMATA FIT (yalnız train)")
print("=" * 60)

model = ProbabilisticAutomata(config)
model.fit(train_symbols)

print(f"  State sayısı: {model.n_states}")
print(f"  Geçiş matrisi şekli: {model.transition_matrix_shape}")
print(f"  Model: {model}")

# ============================================================
# 5. Add-k öncesi / sonrası sıfır kontrolü
# ============================================================
print("\n" + "=" * 60)
print("[4] ADD-K SMOOTHING SIFIR KONTROLÜ")
print("=" * 60)

# Smoothing öncesi ham matrisi manuel hesapla (raporlama için)
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

print(f"  Toplam olası geçiş çifti: {M} × {M} = {M*M}")
print(f"  Gözlemlenen geçiş sayısı: {len(raw_counts)}")
print(f"  Smoothing ÖNCE sıfır sayısı: {zeros_before}  (beklenir > 0)")
print(f"  Smoothing SONRA sıfır sayısı: {zeros_after}  (OLMAMALI)")
assert zeros_after == 0, "HATA: Add-k sonrası hala sıfır var!"
print(f"  ✓ Add-k smoothing sonrası matris tamamen sıfırsız")

# ============================================================
# 6. Bir pencere için log + normalize path-prob + güven skoru
# ============================================================
print("\n" + "=" * 60)
print("[5] BİR PENCERE ÖRNEĞİ: LOG PATH-PROB + GÜVEN SKORU")
print("=" * 60)

# Train patterns üzerinden ilk geçiş çiftini ele al
example_window_symbols = train_symbols[:automata_cfg["paa_segments"]]
print(f"  Örnek pencere sembolleri: {example_window_symbols}")

# Pencere başına log-prob
log_p = model.window_log_prob(example_window_symbols, normalize=True)
conf = model.confidence_score(log_p)

print(f"  Normalize log-prob (per-transition avg): {log_p:.6f}")
print(f"  Güven skoru [0,1]: {conf:.4f}  (yüksek = normal)")
print(f"  Karar: {'NORMAL' if log_p >= model.threshold_ else 'ANOMALİ'}")

# ============================================================
# 7. Train'den hesaplanan eşik değeri
# ============================================================
print("\n" + "=" * 60)
print("[6] TRAIN EŞİĞİ")
print("=" * 60)

print(f"  anomaly_percentile: {model.anomaly_percentile}")
print(f"  Train normalize log-prob → min: {model.train_scores_.min():.6f}")
print(f"  Train normalize log-prob → mean: {model.train_scores_.mean():.6f}")
print(f"  Train normalize log-prob → max: {model.train_scores_.max():.6f}")
print(f"  Eşik ({model.anomaly_percentile}. percentile): {model.threshold_:.6f}")

# ============================================================
# 8. Örnek pencere kararları (test verisi)
# ============================================================
print("\n" + "=" * 60)
print("[7] ÖRNEK PENCERE KARARLARI (TEST)")
print("=" * 60)

labels, raw_scores = model.predict(test_symbols, return_scores=True)
proba = model.predict_proba(test_symbols)

n_show = min(10, len(labels))
print(f"  {'İndex':<8} {'NormLogProb':>14} {'AnomalProba':>13} {'Karar':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*13} {'-'*10}")
for idx in range(n_show):
    karar = "ANOMALİ" if labels[idx] == 1 else "normal"
    print(
        f"  {idx:<8} {raw_scores[idx]:>14.6f} {proba[idx]:>13.4f} {karar:>10}"
    )

# Son birkaç pencere (anomali bölgesi)
print(f"\n  ... (son {n_show} pencere — anomali bölgesi)")
print(f"  {'İndex':<8} {'NormLogProb':>14} {'AnomalProba':>13} {'Karar':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*13} {'-'*10}")
for idx in range(max(0, len(labels) - n_show), len(labels)):
    karar = "ANOMALİ" if labels[idx] == 1 else "normal"
    print(
        f"  {idx:<8} {raw_scores[idx]:>14.6f} {proba[idx]:>13.4f} {karar:>10}"
    )

n_anomaly = int(labels.sum())
print(f"\n  Toplam pencere: {len(labels)}, Tespit edilen anomali: {n_anomaly}")

# ============================================================
# ÖZET
# ============================================================
print("\n" + "=" * 60)
print("ÖZET")
print("=" * 60)
print(f"  SAX alfabe büyüklüğü       : {automata_cfg['alphabet_size']}")
print(f"  PAA segment sayısı         : {automata_cfg['paa_segments']}")
print(f"  Sliding window_size        : {automata_cfg['window_size']}")
print(f"  Smoothing_k                : {automata_cfg['smoothing_k']}")
print(f"  Anomaly percentile         : {automata_cfg['anomaly_percentile']}")
print(f"  Train SAX sembolleri       : {len(train_symbols)}")
print(f"  Test  SAX sembolleri       : {len(test_symbols)}")
print(f"  State (unique pattern) sayısı: {model.n_states}")
print(f"  Geçiş matrisi şekli        : {model.transition_matrix_shape}")
print(f"  Smoothing öncesi sıfır     : {zeros_before}  ✓ (var — normal)")
print(f"  Smoothing sonrası sıfır    : {zeros_after}   ✓ (hiç yok)")
print(f"  Eşik değeri                : {model.threshold_:.6f}")
print(f"  Leakage testi              : GEÇTI ✓")
print("=" * 60)
print("TÜM TESTLER BAŞARILI ✓")
