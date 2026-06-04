"""
automata.py — Olasılıksal Sonlu Otomata (PSA) modeli.

Pipeline özeti:
  1. SAX dizisi üzerinde sliding window (window_size) → her pencere = pattern = state.
  2. Train'den: benzersiz pattern sözlüğü + ardışık pattern geçiş frekansları.
  3. Add-k (Lidstone) smoothing:
       a_ij = (C(i→j) + k) / (Σ_j C(i→j) + k × M)
     k > 0 garantisi → matriste HİÇ sıfır olmaz.
  4. Geçiş matrisi LOG-SPACE'e alınır → underflow engeli.
  5. Bir pencere dizisi için path probability:
       log_path = Σ log(a_{p_i, p_{i+1}})   [log toplamı, çarpım değil]
       normalize  = log_path / n_transitions   [per-transition avg log-likelihood]
  6. Güven skoru: normalize log-prob → [0,1] min-max (train referansıyla).
     Yüksek güven = daha NORMAL; düşük güven = daha ANOMALİK.
  7. Anomali eşiği: train normalize path-prob dağılımının anomaly_percentile.
     Normalize log-prob < threshold → 1 (anomali).
  8. Karar hizalaması: last-step (DL ile simetri).
  9. Sklearn-benzeri: fit(train_symbols) → predict/predict_proba(test_symbols).

KRİTİK leakage koruması:
  - SAX sözlüğü + geçiş matrisi YALNIZCA fit() içinde train verisiyle oluşur.
  - predict/predict_proba state_index_/trans_matrix_'i HİÇ değiştirmez.
  - Add-k sonrası matriste sıfır olmaması assert ile garanti edilir.

Tüm parametreler config['automata'] altından okunur:
  window_size        : sliding window (pattern) uzunluğu
  smoothing_k        : Add-k Lidstone smoothing sabiti (örn. 0.1)
  anomaly_percentile : train log-prob dağılımının kaçıncı percentile'ı eşik (örn. 5)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar (modül içi, private)
# ---------------------------------------------------------------------------

def _symbols_to_patterns(symbols: list[str], window_size: int) -> list[str]:
    """
    SAX sembol listesini sliding window ile pattern listesine dönüştürür.

    Her pencere window_size kadar ardışık sembolü birleştirerek tek bir string
    pattern oluşturur. Pencereler birer adım kaydırılır (stride=1).

    Parametreler
    ------------
    symbols : list[str]
        SAX sembol dizisi (uzunluk L). Her eleman tek karakterlik string.
    window_size : int
        Her pattern'ın uzunluğu (sembol sayısı).

    Döndürür
    --------
    patterns : list[str]
        Birleştirilmiş string pattern listesi.
        Uzunluk = max(0,  L - window_size + 1).
        Örn. symbols=['a','b','a','c'], window_size=2 → ['ab', 'ba', 'ac']

    Yükseltir
    ---------
    ValueError : window_size <= 0 ise.
    """
    if window_size <= 0:
        raise ValueError(f"window_size pozitif olmalı, alınan: {window_size}")
    patterns = []
    for i in range(len(symbols) - window_size + 1):
        pattern = "".join(symbols[i: i + window_size])
        patterns.append(pattern)
    return patterns


def _build_transition_counts(
    patterns: list[str],
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """
    Pattern listesinden state index sözlüğü ve ardışık geçiş sayım tablosu üretir.

    Sıra koruyarak tekil pattern sözlüğü oluşturulur; ilk görülme sırasına
    göre indekslenir. Geçiş sayımı: her (patterns[i], patterns[i+1]) çifti için.

    Döndürür
    --------
    state_index : dict[str, int]
        pattern → 0-tabanlı tamsayı indeks.
    counts : dict[(str, str), int]
        (state_i, state_j) → geçiş sayısı (yalnızca gözlemlenen çiftler).
    """
    # Sıra koruyan tekil liste (Python 3.7+ dict insertion-order)
    unique_patterns = list(dict.fromkeys(patterns))
    state_index: dict[str, int] = {p: i for i, p in enumerate(unique_patterns)}

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(len(patterns) - 1):
        src = patterns[i]
        dst = patterns[i + 1]
        counts[(src, dst)] += 1

    return state_index, dict(counts)


def _build_transition_matrix(
    state_index: dict[str, int],
    counts: dict[tuple[str, str], int],
    smoothing_k: float,
) -> np.ndarray:
    """
    Add-k (Lidstone) smoothing uygulanmış geçiş olasılığı matrisi üretir.

    Formül:
        a_ij = (C(i→j) + k) / (Σ_j C(i→j) + k × M)

    Burada M = state sayısı. k > 0 olduğunda her satır toplamı 1 ve
    matriste HİÇ sıfır bulunmaz.

    Parametreler
    ------------
    state_index : dict[str, int]
        pattern → 0-tabanlı indeks sözlüğü.
    counts : dict[(str, str), int]
        Gözlemlenmiş geçiş sayıları (sıfırlar dahil değil).
    smoothing_k : float
        Lidstone smoothing sabiti (k > 0).

    Döndürür
    --------
    trans_matrix : np.ndarray, shape (M, M)
        Satır-normalleştirilmiş olasılık matrisi. Tüm değerler > 0.

    Yükseltir
    ---------
    AssertionError : Smoothing sonrası herhangi bir eleman 0 ise (olmamalı).
    ValueError     : smoothing_k <= 0 ise.
    """
    if smoothing_k <= 0:
        raise ValueError(f"smoothing_k > 0 olmalı, alınan: {smoothing_k}")

    M = len(state_index)

    # Ham sayım matrisi (float, sıfırla başlar)
    count_matrix = np.zeros((M, M), dtype=float)
    for (src, dst), cnt in counts.items():
        i = state_index[src]
        j = state_index[dst]
        count_matrix[i, j] = cnt

    # Add-k smoothing: her hücreye k ekle → sıfır garantisi
    smoothed = count_matrix + smoothing_k            # (M, M)
    row_sums = smoothed.sum(axis=1, keepdims=True)   # (M, 1)
    trans_matrix = smoothed / row_sums               # satır normalize

    # Matematiksel garanti: k > 0 ile sıfır olmamalı
    assert np.all(trans_matrix > 0), (
        "Add-k smoothing sonrası geçiş matrisinde sıfır tespit edildi — "
        "bu olmamalı. smoothing_k değerini kontrol edin."
    )
    return trans_matrix


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------


class ProbabilisticAutomata:
    """
    Olasılıksal Sonlu Otomata (PSA) tabanlı anomali tespitçisi.

    Kullanım
    --------
    >>> model = ProbabilisticAutomata(config)
    >>> model.fit(train_sax_symbols)
    >>> labels = model.predict(test_sax_symbols)
    >>> proba  = model.predict_proba(test_sax_symbols)

    Parametreler (config['automata'])
    ----------------------------------
    window_size        : int   — sliding window (pattern) uzunluğu
    smoothing_k        : float — Add-k Lidstone sabit (> 0)
    anomaly_percentile : int   — train eşiği için alt yüzdelik dilim (örn. 5)

    Dahili durum (fit sonrası dondurulur — data leakage yok)
    ---------------------------------------------------------
    state_index_       : dict[str, int]      — pattern → state indeksi
    _inv_state_index   : dict[int, str]      — state indeksi → pattern (debug)
    trans_matrix_      : np.ndarray (M, M)   — olasılık matrisi (doğrusal)
    _log_trans_matrix  : np.ndarray (M, M)   — log-olasılık matrisi (log-space)
    _raw_counts_       : dict                — smoothing öncesi ham sayımlar (denetim)
    threshold_         : float               — anomali eşiği (normalize log-prob)
    train_scores_      : np.ndarray          — train pencere normalize log-prob'ları
    is_fitted_         : bool

    Karar hizalaması (last-step)
    ----------------------------
    Pencere i'nin kararı, pencerenin SON satırına hizalanır (DL ile simetri).
    predict() çıktısı indeksi: scores[i] → orijinal veri satırı window_size + i - 1.
    """

    def __init__(self, config: dict):
        automata_cfg = config["automata"]
        self.window_size: int = int(automata_cfg["window_size"])
        self.smoothing_k: float = float(automata_cfg["smoothing_k"])
        self.threshold_selection: str = automata_cfg.get("threshold_selection", "validation_f1_bidirectional")

        # Fit sonrası dolan alanlar — None ile başlar (leakage denetim kolaylığı)
        self.state_index_: dict[str, int] = {}
        self._inv_state_index: dict[int, str] = {}
        self.trans_matrix_: np.ndarray | None = None      # (M, M) doğrusal
        self._log_trans_matrix: np.ndarray | None = None  # (M, M) log-space
        self._raw_counts_: dict[tuple[str, str], int] = {}  # smoothing öncesi
        self.threshold_: float | None = None
        self.anomaly_direction_: str | None = None
        self.train_scores_: np.ndarray | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # fit — YALNIZCA train verisi üzerinde çağrılmalı
    # ------------------------------------------------------------------

    def fit(self, train_symbols: list[str]) -> "ProbabilisticAutomata":
        """
        Otomata modelini yalnızca train SAX sembol dizisi üzerinde eğitir.

        Parametreler
        ------------
        train_symbols : list[str]
            Train PC1'in SAX sembol listesi (PAASAXTransformer.transform çıktısı).
            Uzunluk >= window_size + 1 olmalıdır (en az bir geçiş için).

        Adımlar
        -------
        1. Sliding window (window_size) → pattern listesi
        2. Benzersiz pattern sözlüğü (state_index_)
        3. Geçiş sayımları (ham, smoothing öncesi)
        4. Add-k smoothing → trans_matrix_  (tüm değerler > 0)
        5. Log-space matris (_log_trans_matrix)
        6. Train pencerelerinin normalize path-prob'ları (train_scores_)
        (Threshold secimi disaridan set_threshold metoduyla validation setinden yapilir)

        Döndürür
        --------
        self
        """
        min_len = self.window_size + 1
        if len(train_symbols) < min_len:
            raise ValueError(
                f"Train sembol dizisi en az window_size+1={min_len} "
                f"uzunluğunda olmalı, alınan: {len(train_symbols)}"
            )

        # ---- 1. Pattern listesi ----
        patterns = _symbols_to_patterns(train_symbols, self.window_size)
        logger.info(
            "[Automata.fit] %d sembol → %d pattern (window_size=%d)",
            len(train_symbols), len(patterns), self.window_size,
        )

        # ---- 2. State sözlüğü + ham geçiş sayımları ----
        self.state_index_, self._raw_counts_ = _build_transition_counts(patterns)
        self._inv_state_index = {v: k for k, v in self.state_index_.items()}
        M = len(self.state_index_)
        logger.info("[Automata.fit] %d benzersiz state (pattern) bulundu.", M)

        # ---- 3. Add-k smoothing ile geçiş matrisi ----
        self.trans_matrix_ = _build_transition_matrix(
            self.state_index_, self._raw_counts_, self.smoothing_k
        )

        # ---- 4. Log-space matris ----
        # Add-k garanti ettiğinden log(0) = -inf riski yoktur
        self._log_trans_matrix = np.log(self.trans_matrix_)

        # ---- 5. Train normalize path-prob'ları ----
        self.train_scores_ = self._compute_path_scores(patterns)
        logger.info(
            "[Automata.fit] %d train geçiş skoru hesaplandı.", len(self.train_scores_)
        )

        self.is_fitted_ = True
        return self
        
    def set_threshold(self, threshold: float, direction: str) -> None:
        """
        Validation setinden bulunan en iyi eşiği ve yönü modele kaydeder.
        direction: 'low' (düşük log-prob = anomali) veya 'high' (yüksek log-prob = anomali)
        """
        self.threshold_ = threshold
        self.anomaly_direction_ = direction
        logger.info(f"[Automata.set_threshold] Eşik: {threshold:.6f}, Yön: {direction.upper()}")

    # ------------------------------------------------------------------
    # Dahili: pattern geçiş dizisinden normalize path-prob dizisi
    # ------------------------------------------------------------------

    def _compute_path_scores(self, patterns: list[str]) -> np.ndarray:
        """
        Ardışık pattern çiftleri için LOG-SPACE **per-transition** normalize
        path-probability hesaplar.

        Bir geçiş adımı için log-prob:
            log(a_{p_i, p_{i+1}})

        Birden fazla adım içeren bir yol için:
            log_path = Σ_i log(a_{p_i, p_{i+1}})
            normalize = log_path / n_transitions

        Bu implementasyonda her çift (p_i, p_{i+1}) ayrı bir skor olarak döner;
        yani score[k] = log(a_{p_k, p_{k+1}}) / 1 = log(a_{p_k, p_{k+1}}).
        Bu, per-transition average log-likelihood ile tutarlıdır.

        Bilinmeyen pattern (unseen) için:
            log( 1 / M )   ←  uniform olasılık varsayımı (en kötü durum)

        Parametreler
        ------------
        patterns : list[str]
            Sliding window ile üretilen pattern listesi.

        Döndürür
        --------
        scores : np.ndarray, shape (len(patterns) - 1,)
            Her ardışık (p_i, p_{i+1}) çifti için per-transition log-prob.
            Yüksek değer = daha NORMAL geçiş.
        """
        M = len(self.state_index_)
        log_prob_unseen = np.log(1.0 / M) if M > 0 else -np.inf

        scores: list[float] = []
        for i in range(len(patterns) - 1):
            src = patterns[i]
            dst = patterns[i + 1]
            src_idx = self.state_index_.get(src)
            dst_idx = self.state_index_.get(dst)

            if src_idx is None or dst_idx is None:
                # Unseen state → en kötü olası log-prob
                scores.append(log_prob_unseen)
            else:
                scores.append(float(self._log_trans_matrix[src_idx, dst_idx]))

        return np.array(scores, dtype=float)

    # ------------------------------------------------------------------
    # Dahili: tam SAX sembol dizisini sliding window ile tarayan skor hesabı
    # ------------------------------------------------------------------

    def _score_symbols(self, symbols: list[str]) -> np.ndarray:
        """
        SAX sembol listesini sliding window + pattern geçiş skoru ile tarar.

        Döndürür
        --------
        scores : np.ndarray, shape (n_patterns - 1,)
            Her ardışık pattern geçişi için per-transition normalize log-prob.

        Notlar
        ------
        - predict()/predict_proba() bu metodu çağırır; state_index_/trans_matrix_
          değiştirilmez (data leakage yok).
        - Uzunluk < window_size + 1 ise boş dizi döner.
        """
        if not self.is_fitted_:
            raise RuntimeError("predict/score çağrılmadan önce fit() gereklidir.")

        patterns = _symbols_to_patterns(symbols, self.window_size)
        if len(patterns) < 2:
            return np.array([], dtype=float)

        return self._compute_path_scores(patterns)

    # ------------------------------------------------------------------
    # predict_proba — [0,1] anomali olasılığı
    # ------------------------------------------------------------------

    def predict_proba(self, test_symbols: list[str]) -> np.ndarray:
        """
        Her pencere geçişi için [0,1] aralığında anomali olasılığı döndürür.

        Dönüşüm (train referansıyla min-max ölçekleme):
            normalized = (score - train_min) / (train_max - train_min)
            anomaly_proba = 1 - clip(normalized, 0, 1)

        Normalize log-prob ne kadar DÜŞÜKSE anomali o kadar YÜKSEK.

        Parametreler
        ------------
        test_symbols : list[str]
            Test SAX sembol listesi.

        Döndürür
        --------
        proba : np.ndarray, shape (n_patterns - 1,)
            Anomali olasılığı; 1.0 = en anomalik, 0.0 = en normal.
        """
        raw_scores = self._score_symbols(test_symbols)
        if len(raw_scores) == 0:
            return np.array([], dtype=float)

        train_min = float(self.train_scores_.min())
        train_max = float(self.train_scores_.max())
        span = (train_max - train_min) if (train_max - train_min) > 1e-12 else 1.0

        normalized = (raw_scores - train_min) / span
        normalized = np.clip(normalized, 0.0, 1.0)
        anomaly_proba = 1.0 - normalized  # ters çevir: yüksek = anomali

        return anomaly_proba

    # ------------------------------------------------------------------
    # predict — binary etiket dizisi (0=normal, 1=anomali)
    # ------------------------------------------------------------------

    def predict(
        self,
        test_symbols: list[str],
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Test SAX sembol dizisi için anomali kararı verir (last-step hizalaması).

        Karar mantığı:
            direction 'low'  ise: log-prob < threshold_  →  1 (anomali)
            direction 'high' ise: log-prob > threshold_  →  1 (anomali)

        Parametreler
        ------------
        test_symbols : list[str]
            Test SAX sembol listesi.
        return_scores : bool
            True ise (labels, raw_scores) döndürür; False ise yalnızca labels.

        Döndürür
        --------
        labels : np.ndarray, shape (n_patterns - 1,)
            0 (normal) veya 1 (anomali).
        scores : np.ndarray, shape (n_patterns - 1,)   [return_scores=True ise]
            Per-transition normalize log-prob (düşük = daha anomalik).
        """
        raw_scores = self._score_symbols(test_symbols)
        if len(raw_scores) == 0:
            empty_labels = np.array([], dtype=int)
            if return_scores:
                return empty_labels, np.array([], dtype=float)
            return empty_labels

        if self.threshold_ is None or self.anomaly_direction_ is None:
            raise ValueError("Threshold ve yön set edilmedi! Lütfen predict()'ten önce set_threshold() çağırın.")

        if self.anomaly_direction_ == 'low':
            labels = (raw_scores < self.threshold_).astype(int)
        else:
            labels = (raw_scores > self.threshold_).astype(int)

        if return_scores:
            return labels, raw_scores
        return labels

    # ------------------------------------------------------------------
    # Tek pencere yardımcıları (raporlama / debug)
    # ------------------------------------------------------------------

    def window_log_prob(self, window_patterns: list[str], normalize: bool = True) -> float:
        """
        Verilen ardışık pattern listesi için log-prob döndürür.

        Parametreler
        ------------
        window_patterns : list[str]
            Ardışık state (pattern) listesi; en az 2 eleman gerekli.
            Her eleman `window_size` uzunluğundaki bir SAX pattern string'i.
        normalize : bool
            True ise geçiş sayısına böl (per-transition avg log-likelihood).

        Döndürür
        --------
        float
            Log-prob değeri. Yüksek = daha normal geçiş.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit() çağrılmadan kullanılamaz.")
        if len(window_patterns) < 2:
            return float("nan")

        M = len(self.state_index_)
        log_prob_unseen = np.log(1.0 / M) if M > 0 else -np.inf

        log_probs: list[float] = []
        for i in range(len(window_patterns) - 1):
            src = window_patterns[i]
            dst = window_patterns[i + 1]
            src_idx = self.state_index_.get(src)
            dst_idx = self.state_index_.get(dst)
            if src_idx is None or dst_idx is None:
                log_probs.append(log_prob_unseen)
            else:
                log_probs.append(float(self._log_trans_matrix[src_idx, dst_idx]))

        total = float(np.sum(log_probs))
        return total / len(log_probs) if normalize else total

    def confidence_score(self, normalize_log_prob: float) -> float:
        """
        Tek bir normalize log-prob değerinden [0,1] güven skoru türetir.

        Yüksek değer = daha NORMAL (güvenilir geçiş).
        Düşük değer  = daha ANOMALİK.

        Dönüşüm: (score - train_min) / (train_max - train_min) → clip [0,1].
        """
        if self.train_scores_ is None:
            raise RuntimeError("fit() çağrılmadan kullanılamaz.")
        train_min = float(self.train_scores_.min())
        train_max = float(self.train_scores_.max())
        span = (train_max - train_min) if (train_max - train_min) > 1e-12 else 1.0
        return float(np.clip((normalize_log_prob - train_min) / span, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Sıfır / leakage denetim araçları
    # ------------------------------------------------------------------

    def has_zeros_before_smoothing(self) -> bool:
        """
        Smoothing öncesinde geçiş matrisinde sıfır olup olmadığını döndürür.

        Ham sayım matrisi M×M boyutundadır; gözlemlenmemiş geçiş çiftleri
        sıfır olacağından tipik durum True'dur.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit() çağrılmadan kullanılamaz.")
        M = self.n_states
        observed = len(self._raw_counts_)
        # M*M hücrenin tamamı gözlemlenmişse sıfır yok; aksi halde var
        return observed < M * M

    def has_zeros_after_smoothing(self) -> bool:
        """
        Smoothing sonrası matriste sıfır var mı? (Olmamalı — Add-k garantisi.)
        """
        if self.trans_matrix_ is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        return bool(np.any(self.trans_matrix_ == 0))

    # ------------------------------------------------------------------
    # Bilgi sorgulama property'leri
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        """Benzersiz state (pattern) sayısı."""
        return len(self.state_index_)

    @property
    def transition_matrix_shape(self) -> tuple[int, int]:
        """Geçiş matrisinin şekli (M, M)."""
        if self.trans_matrix_ is None:
            return (0, 0)
        return self.trans_matrix_.shape

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        if self.is_fitted_:
            return (
                f"ProbabilisticAutomata("
                f"n_states={self.n_states}, "
                f"window_size={self.window_size}, "
                f"smoothing_k={self.smoothing_k}, "
                f"threshold={self.threshold_:.4f}, "
                f"status={status})"
            )
        return (
            f"ProbabilisticAutomata("
            f"window_size={self.window_size}, "
            f"smoothing_k={self.smoothing_k}, "
            f"status={status})"
        )
