"""
test_automata_pipeline.py — PAA-SAX + Olasılıksal Otomata butunlesik testi.

Calistirma:
    python tests/test_automata_pipeline.py

Raporlanan ciktilar:
  1. SAX dizisi ornegi (train ve test)
  2. State (pattern) sayisi
  3. Gecis matrisi sekli
  4. Add-k oncesi/sonrasi sifir durumu
  5. Ornek pencere icin log + normalize path-prob ve guven skoru
  6. Train'den hesaplanan esik degeri
  7. Ornek pencere kararlari
  8. Fit/transform leakage testi sonucu
"""

import sys
import os
import io
import numpy as np

# Windows terminal encoding sorununu coz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Proje kokunu path'e ekle (tests/ klasorunden calisirken)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.models.automata.paa_sax import PAASAXTransformer
from src.models.automata.automata import ProbabilisticAutomata, _symbols_to_patterns


# ---------------------------------------------------------------------------
# Yardimci: bolum basligı
# ---------------------------------------------------------------------------
def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Sentetik veri uretimi
# ---------------------------------------------------------------------------
def make_synthetic_signals(rng: np.random.Generator, n_train: int = 300, n_test: int = 100):
    """
    Deterministik sentetik PC1 sinyali uretir.
    Train: normal sinuzoidal sinyal.
    Test:  normal sinyal + ortasinda anomali (yuksek gurultu).
    """
    t_train = np.linspace(0, 4 * np.pi, n_train)
    train_signal = np.sin(t_train) + 0.1 * rng.standard_normal(n_train)

    t_test = np.linspace(0, 2 * np.pi, n_test)
    test_signal = np.sin(t_test) + 0.1 * rng.standard_normal(n_test)
    # Ortaya anomali enjekte et (40-60 indeksleri)
    test_signal[40:60] += 3.0 * rng.standard_normal(20)

    return train_signal, test_signal


# ---------------------------------------------------------------------------
# Ana test rutini
# ---------------------------------------------------------------------------
def run_test() -> bool:
    all_passed = True

    # -----------------------------------------------------------------------
    header("0. KONFIGURASYON YUKLEME")
    # -----------------------------------------------------------------------
    config = load_config(os.path.join(PROJECT_ROOT, "config", "config.yaml"))
    auto_cfg = config["automata"]
    print(f"  alphabet_size      = {auto_cfg['alphabet_size']}")
    print(f"  paa_segments       = {auto_cfg['paa_segments']}")
    print(f"  window_size        = {auto_cfg['window_size']}")
    print(f"  smoothing_k        = {auto_cfg['smoothing_k']}")
    print(f"  anomaly_percentile = {auto_cfg['anomaly_percentile']}")

    # -----------------------------------------------------------------------
    header("1. SENTETIK VERI")
    # -----------------------------------------------------------------------
    seed = config["seeds"][0]
    rng = np.random.default_rng(seed)
    train_signal, test_signal = make_synthetic_signals(rng, n_train=300, n_test=100)
    print(f"  Train sinyal uzunlugu : {len(train_signal)}")
    print(f"  Test  sinyal uzunlugu : {len(test_signal)}")
    expected_train_sax = len(train_signal) - auto_cfg['paa_segments'] + 1
    expected_test_sax  = len(test_signal)  - auto_cfg['paa_segments'] + 1
    print(f"  Beklenen train SAX sembol sayisi : {expected_train_sax}")
    print(f"  Beklenen test  SAX sembol sayisi : {expected_test_sax}")

    # -----------------------------------------------------------------------
    header("2. PAA-SAX FIT (yalnizca train)")
    # -----------------------------------------------------------------------
    transformer = PAASAXTransformer(config)
    train_sax = transformer.fit_transform(train_signal)
    print(f"  Train SAX sembol sayisi : {len(train_sax)}")
    print(f"  Train SAX ornegi (ilk 30): {''.join(train_sax[:30])}")
    print(f"  Benzersiz semboller      : {sorted(set(train_sax))}")

    bps = transformer.get_breakpoints()
    if bps is not None:
        print(f"  Breakpoint degerleri ({len(bps)} adet): {np.round(bps, 6)}")
    else:
        print("  Breakpoint degerleri: pyts bin_edges_ attribute'u mevcut degil.")

    # Sembol sayisi kontrolu
    if len(train_sax) == expected_train_sax:
        ok(f"Train SAX sembol sayisi dogru: {len(train_sax)}")
    else:
        fail(f"Train SAX sembol sayisi beklenenden farkli: {len(train_sax)} != {expected_train_sax}")
        all_passed = False

    # -----------------------------------------------------------------------
    header("3. SAX TRANSFORM — TEST (donmus breakpoint'ler)")
    # -----------------------------------------------------------------------
    test_sax = transformer.transform(test_signal)
    print(f"  Test SAX sembol sayisi  : {len(test_sax)}")
    print(f"  Test SAX ornegi (ilk 30): {''.join(test_sax[:30])}")
    print(f"  Benzersiz semboller     : {sorted(set(test_sax))}")

    if len(test_sax) == expected_test_sax:
        ok(f"Test SAX sembol sayisi dogru: {len(test_sax)}")
    else:
        fail(f"Test SAX sembol sayisi beklenenden farkli: {len(test_sax)} != {expected_test_sax}")
        all_passed = False

    # -----------------------------------------------------------------------
    header("4. FIT/TRANSFORM LEAKAGE TESTI")
    # -----------------------------------------------------------------------
    bps_before_ids = id(transformer._sax)   # SAX nesne kimlik kontrolu

    # Cok uzak degerli test sinyali uygula — breakpoint'ler degismemeli
    extreme_test = np.full(len(test_signal), 1000.0)
    _ = transformer.transform(extreme_test)

    bps_after_ids = id(transformer._sax)

    # 1. SAX nesnesi degismemeli (yeni fit() olmamali)
    if bps_before_ids == bps_after_ids:
        ok("SAX nesnesi transform sonrasi DEGISMEDI -> leakage YOK")
    else:
        fail("SAX nesnesi transform sonrasi DEGISTI -> LEAKAGE!")
        all_passed = False

    # 2. assert_no_leakage ile dogrula
    try:
        transformer.assert_no_leakage(test_signal)
        ok("assert_no_leakage gecti -> breakpoint'ler sabit")
    except AssertionError as e:
        fail(f"assert_no_leakage BASARISIZ: {e}")
        all_passed = False

    # -----------------------------------------------------------------------
    header("5. OTOMATA FIT (yalnizca train SAX)")
    # -----------------------------------------------------------------------
    model = ProbabilisticAutomata(config)
    model.fit(train_sax)
    print(f"  State (pattern) sayisi   : {model.n_states}")
    print(f"  Gecis matrisi sekli       : {model.transition_matrix_shape}")
    print(f"  Esik degeri (threshold)  : {model.threshold_:.6f}")
    print(f"  Egitildi mi              : {model.is_fitted_}")
    print(f"  Model repr               : {model!r}")

    # -----------------------------------------------------------------------
    header("6. ADD-K SMOOTHING SIFIR KONTROLU")
    # -----------------------------------------------------------------------
    zeros_before = model.has_zeros_before_smoothing()
    zeros_after  = model.has_zeros_after_smoothing()
    print(f"  Ham gecis cift sayisi      : {len(model._raw_counts_)}")
    print(f"  Toplam olasilı cift (M^2)  : {model.n_states ** 2}")
    print(f"  Smoothing ONCESI sifir var mi : {zeros_before}  (tipik: True)")
    print(f"  Smoothing SONRASI sifir var mi : {zeros_after}  (olmamali: False)")

    if zeros_after:
        fail("Smoothing sonrasi sifir tespit edildi — hata!")
        all_passed = False
    else:
        ok("Smoothing sonrasi sifir YOK -> Add-k garantisi saglandi")

    # Matris satir toplamlarini kontrol et (her satir = 1.0)
    row_sums = model.trans_matrix_.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-10):
        ok("Gecis matrisi satir toplamları 1.0 -> olasilik dagilimi gecerli")
    else:
        fail(f"Satir toplamlari 1.0 degil: min={row_sums.min():.8f}, max={row_sums.max():.8f}")
        all_passed = False

    # -----------------------------------------------------------------------
    header("7. ORNEK PENCERE LOG-PROB VE GUVEN SKORU")
    # -----------------------------------------------------------------------
    train_patterns = _symbols_to_patterns(train_sax, model.window_size)
    print(f"  Train pattern sayisi : {len(train_patterns)}")
    print(f"  Ornek patternler (ilk 5): {train_patterns[:5]}")

    if len(train_patterns) >= 2:
        example_pair = [train_patterns[0], train_patterns[1]]
        log_prob_norm  = model.window_log_prob(example_pair, normalize=True)
        log_prob_raw   = model.window_log_prob(example_pair, normalize=False)
        conf_score     = model.confidence_score(log_prob_norm)

        print(f"\n  Ornek pattern cifti    : {example_pair}")
        print(f"  Log-prob (ham toplam)  : {log_prob_raw:.6f}")
        print(f"  Log-prob (normalize)   : {log_prob_norm:.6f}  (per-transition avg)")
        print(f"  Guven skoru [0,1]      : {conf_score:.6f}  (1=normal, 0=anomali)")

        # Guven skoru gecerlilik araligi
        if 0.0 <= conf_score <= 1.0:
            ok("Guven skoru [0,1] araliginda")
        else:
            fail(f"Guven skoru aralik disi: {conf_score}")
            all_passed = False
    else:
        print("  Yeterli pattern yok (window_size cok buyuk?).")

    # -----------------------------------------------------------------------
    header("8. TRAIN ANOMALİ ESIGI")
    # -----------------------------------------------------------------------
    print(f"  anomaly_percentile : {model.anomaly_percentile}")
    print(f"  Train esigi        : {model.threshold_:.6f}")
    print(f"  Train scores min   : {model.train_scores_.min():.6f}")
    print(f"  Train scores max   : {model.train_scores_.max():.6f}")
    print(f"  Train scores mean  : {model.train_scores_.mean():.6f}")
    print(f"  Train scores std   : {model.train_scores_.std():.6f}")

    # Esik, min ve max arasinda olmali
    if model.train_scores_.min() <= model.threshold_ <= model.train_scores_.max():
        ok("Esik degeri train scores araliginda")
    else:
        fail(f"Esik degeri beklenen aralik disinda!")
        all_passed = False

    # -----------------------------------------------------------------------
    header("9. PREDICT VE PREDICT_PROBA (test sinyali)")
    # -----------------------------------------------------------------------
    labels, raw_scores = model.predict(test_sax, return_scores=True)
    proba              = model.predict_proba(test_sax)

    print(f"  Test SAX uzunlugu             : {len(test_sax)}")
    print(f"  Test pencere gecis sayisi     : {len(labels)}")
    print(f"  Anomali karari (ilk 20)       : {list(labels[:20])}")
    print(f"  Raw log scores (ilk 10)       : {[round(float(s),4) for s in raw_scores[:10]]}")
    print(f"  Anomali olasıligi (ilk 10)    : {[round(float(p),4) for p in proba[:10]]}")
    print(f"  Toplam anomali karari         : {labels.sum()} / {len(labels)}")
    print(f"  Anomali orani                 : {labels.mean():.2%}")

    # Proba aralik kontrolu
    if np.all((proba >= 0.0) & (proba <= 1.0)):
        ok("predict_proba degerleri [0,1] araliginda")
    else:
        fail(f"predict_proba deger aralik disi: min={proba.min():.4f}, max={proba.max():.4f}")
        all_passed = False

    # -----------------------------------------------------------------------
    header("10. OTOMATA FIT/PREDICT LEAKAGE TESTI")
    # -----------------------------------------------------------------------
    state_idx_before = dict(model.state_index_)
    trans_before     = model.trans_matrix_.copy()
    log_trans_before = model._log_trans_matrix.copy()
    threshold_before = model.threshold_

    # predict() ve predict_proba() state'i degistirmemeli
    _ = model.predict(test_sax)
    _ = model.predict_proba(test_sax)

    state_idx_after  = dict(model.state_index_)
    trans_after      = model.trans_matrix_
    log_trans_after  = model._log_trans_matrix
    threshold_after  = model.threshold_

    leakage_detected = False
    if state_idx_before != state_idx_after:
        fail("state_index_ predict sonrasi DEGISTI -> LEAKAGE!")
        leakage_detected = True; all_passed = False

    if not np.allclose(trans_before, trans_after):
        fail("trans_matrix_ predict sonrasi DEGISTI -> LEAKAGE!")
        leakage_detected = True; all_passed = False

    if not np.allclose(log_trans_before, log_trans_after):
        fail("_log_trans_matrix predict sonrasi DEGISTI -> LEAKAGE!")
        leakage_detected = True; all_passed = False

    if threshold_before != threshold_after:
        fail("threshold_ predict sonrasi DEGISTI -> LEAKAGE!")
        leakage_detected = True; all_passed = False

    if not leakage_detected:
        ok("state_index_, trans_matrix_, threshold_ predict sonrasi DEGISMEDI -> leakage YOK")

    # -----------------------------------------------------------------------
    header("11. SINIR DURUM TESTLERI")
    # -----------------------------------------------------------------------

    # fit() cagirilmadan transform() -> RuntimeError
    t2 = PAASAXTransformer(config)
    try:
        t2.transform(test_signal)
        fail("fit() olmadan transform() cagrildi, RuntimeError bekleniyor!")
        all_passed = False
    except RuntimeError:
        ok("fit() oncesi transform() dogru sekilde RuntimeError firlatti")

    # fit() cagirilmadan predict() -> RuntimeError
    m2 = ProbabilisticAutomata(config)
    try:
        m2.predict(test_sax)
        fail("fit() olmadan predict() cagrildi, RuntimeError bekleniyor!")
        all_passed = False
    except RuntimeError:
        ok("fit() oncesi predict() dogru sekilde RuntimeError firlatti")

    # Cok kisa sembol dizisi fit() -> ValueError
    try:
        m2.fit(["a"])   # window_size+1 gerekli
        fail("Kisa dizi fit() kabul etti, ValueError bekleniyor!")
        all_passed = False
    except ValueError:
        ok("Kisa sembol dizisi fit() dogru sekilde ValueError firlatti")

    # SAX cok kisa sinyal -> ValueError
    t3 = PAASAXTransformer(config)
    try:
        t3.fit(np.array([0.1]))   # paa_segments kadar nokta yok
        fail("Kisa sinyal transform kabul edildi, ValueError bekleniyor!")
        all_passed = False
    except ValueError:
        ok("Kisa sinyal fit() dogru sekilde ValueError firlatti")

    # -----------------------------------------------------------------------
    header("12. LOG-SPACE UNDERFLOW KORUMASI KONTROLU")
    # -----------------------------------------------------------------------
    # Tum log-prob degerleri sonlu (nan/inf degil) olmali
    if np.all(np.isfinite(raw_scores)):
        ok(f"Tum test log-prob degerleri sonlu (nan/inf YOK) - {len(raw_scores)} deger")
    else:
        n_nan = np.sum(~np.isfinite(raw_scores))
        fail(f"{n_nan} adet nan/inf tespit edildi — log-space hesabinda sorun var!")
        all_passed = False

    if np.all(np.isfinite(model.train_scores_)):
        ok(f"Tum train log-prob degerleri sonlu - {len(model.train_scores_)} deger")
    else:
        n_nan = np.sum(~np.isfinite(model.train_scores_))
        fail(f"Train scores'ta {n_nan} adet nan/inf!")
        all_passed = False

    # -----------------------------------------------------------------------
    header("SONUC")
    # -----------------------------------------------------------------------
    if all_passed:
        print("\n  *** TUM TESTLER BASARILI ***")
        print()
    else:
        print("\n  XXX BAZI TESTLER BASARISIZ — yukaridaki hatalari kontrol edin XXX")
        print()

    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
