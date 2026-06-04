"""
paa_sax.py — Sliding-window + quantile-based SAX dönüşümü.

pyts 0.13.0 NOT: pyts.approximation.SymbolicAggregateApproximation stateless
çalışır — fit() hiçbir şey kaydetmez, transform() her seferinde breakpoint'leri
yeniden hesaplar. Bu, projemizin "train'den öğren, dondur" kuralıyla uyumsuz.

ÇÖZÜM: Breakpoint'leri doğrudan np.percentile ile train sliding-window
matrisinden hesaplayıp _breakpoints attribute'unda saklıyoruz.
transform() bu donmuş breakpoint'lerle np.digitize yaparak sembolleştirir.
pyts'in SAX sınıfı artık KULLANILMIYOR — breakpoint hesabı saf NumPy.

Mimari:
  1. Sinyal sliding window ile (n_windows, paa_segments) matrisine dönüştürülür.
  2. fit():  Train matrisinin TÜM elemanları üzerinden quantile breakpoint'ler
             hesaplanır ve dondurulur.
  3. transform(): Donmuş breakpoint'lerle her değeri bin'e atar, karşılık gelen
             alfabe harfine çevirir. Her pencerenin son sembolü alınır (last-step).

Leakage garantisi:
  - _breakpoints yalnızca fit() içinde atanır.
  - transform() breakpoint'leri asla değiştirmez.
  - fit() öncesi transform() çağrılırsa RuntimeError fırlatır.

Parametreler (config['automata']):
  alphabet_size : SAX alfabe büyüklüğü (n_bins)
  paa_segments  : Sliding window boyutu
  sax_strategy  : 'quantile' (config-driven, hard-code yok)
"""

from __future__ import annotations

import numpy as np


class PAASAXTransformer:
    """
    Sliding-window + quantile breakpoint SAX dönüşümü; sklearn-benzeri fit/transform.

    Parametreler
    ------------
    config : dict
        config.yaml içeriği; config['automata']['alphabet_size'],
        config['automata']['paa_segments'] ve config['automata']['sax_strategy']
        anahtarları zorunludur.

    Dahili durum
    ------------
    _breakpoints        : np.ndarray  (n_bins-1 eleman, quantile sınırları)
    _breakpoints_snap   : np.ndarray  (fit anındaki breakpoint kopyası, leakage testi)
    _alphabet           : np.ndarray  (alfabe harfleri, ör. ['a','b','c'])
    _is_fitted          : bool
    """

    def __init__(self, config: dict):
        auto = config["automata"]
        self.alphabet_size: int = int(auto["alphabet_size"])
        self.paa_segments: int  = int(auto["paa_segments"])
        self.strategy: str      = str(auto.get("sax_strategy", "quantile"))
        self._is_fitted: bool = False
        self._breakpoints: np.ndarray | None = None
        self._breakpoints_snap: np.ndarray | None = None
        self._alphabet: np.ndarray = np.array(
            [chr(i) for i in range(97, 97 + self.alphabet_size)]
        )

    # ------------------------------------------------------------------
    # Dahili: sliding-window matris oluştur
    # ------------------------------------------------------------------

    def _make_windows(self, signal: np.ndarray) -> np.ndarray:
        """
        1D sinyali (N,) sliding-window matrisi (n_windows, paa_segments)'e çevirir.

        n_windows = N - paa_segments + 1.
        Her satır W = paa_segments zamanlı ardışık pencere.
        """
        sig = np.asarray(signal, dtype=float).ravel()
        N, W = len(sig), self.paa_segments
        if N < W:
            raise ValueError(
                f"Sinyal uzunluğu ({N}) paa_segments'ten ({W}) küçük olamaz."
            )
        windows = np.lib.stride_tricks.sliding_window_view(sig, W).copy()
        return windows

    # ------------------------------------------------------------------
    # Dahili: breakpoint hesaplama (quantile / uniform / normal)
    # ------------------------------------------------------------------

    def _compute_breakpoints(self, data_flat: np.ndarray) -> np.ndarray:
        """
        1D veri dizisinden n_bins-1 adet breakpoint hesaplar.

        strategy='quantile' → np.percentile ile eşit-yoğunluklu sınırlar.
        strategy='uniform'  → min-max arasında eşit aralıklı sınırlar.
        strategy='normal'   → standart normal dağılım sınırları (veri-bağımsız).

        Döndürür
        --------
        breakpoints : np.ndarray, shape (n_bins - 1,)
        """
        n_bins = self.alphabet_size
        if self.strategy == "quantile":
            percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            bps = np.percentile(data_flat, percentiles)
        elif self.strategy == "uniform":
            mn, mx = data_flat.min(), data_flat.max()
            bps = np.linspace(mn, mx, n_bins + 1)[1:-1]
        elif self.strategy == "normal":
            from scipy.stats import norm as _norm
            bps = _norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
        else:
            raise ValueError(f"Bilinmeyen sax_strategy: {self.strategy}")
        return bps

    # ------------------------------------------------------------------
    # Dahili: breakpoint'lerle sembolleştir
    # ------------------------------------------------------------------

    def _digitize(self, windows: np.ndarray) -> np.ndarray:
        """
        (n_windows, W) matrisini donmuş breakpoint'lerle sembol matrisine çevirir.

        np.digitize ile her değer bir bin indeksine atanır,
        ardından alfabe harfine dönüştürülür.

        Döndürür
        --------
        symbols : np.ndarray, shape (n_windows, W), dtype=str
        """
        indices = np.digitize(windows, self._breakpoints)
        # digitize n_bins adet bin verir: 0..n_bins-1
        # clip: alphabet_size dışına taşma engelle
        indices = np.clip(indices, 0, self.alphabet_size - 1)
        return self._alphabet[indices]

    # ------------------------------------------------------------------
    # fit -- YALNIZCA train
    # ------------------------------------------------------------------

    def fit(self, train_signal: np.ndarray) -> "PAASAXTransformer":
        """
        SAX breakpoint'lerini yalnızca train sinyalinin sliding pencereleri üzerinden öğrenir.

        Parametreler
        ------------
        train_signal : array-like, shape (N,)
            Train PC1 sinyali. N >= paa_segments olmalı.

        Adımlar
        -------
        1. Train sinyal -> sliding-window matrisi (n_train_wins, paa_segments).
        2. Tüm pencere elemanlarını düzleştir (1D).
        3. Breakpoint'leri hesapla (strategy config'den) → dondur.

        Dönüş
        -----
        self
        """
        windows = self._make_windows(train_signal)
        data_flat = windows.ravel()

        self._breakpoints = self._compute_breakpoints(data_flat)
        self._breakpoints_snap = self._breakpoints.copy()
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform -- donmuş breakpoint'lerle herhangi veriye uygula
    # ------------------------------------------------------------------

    def transform(self, signal: np.ndarray) -> list[str]:
        """
        Donmuş breakpoint'lerle sinyali SAX sembol listesine dönüştürür.

        Parametreler
        ------------
        signal : array-like, shape (N,)
            Dönüştürülecek sinyal (train/val/test).

        Dönüş
        -----
        symbols : list[str]
            Uzunluk = N - paa_segments + 1.
            Her eleman tek karakterlik SAX sembol ('a', 'b', 'c', ...).

        Leakage notu
        ------------
        Bu metot _breakpoints'i DEĞİŞTİRMEZ, yalnızca okur.
        """
        if not self._is_fitted:
            raise RuntimeError("transform() öncesi fit() çağrılmalıdır.")

        windows = self._make_windows(signal)
        sym_matrix = self._digitize(windows)  # (n_wins, W)

        # Her pencerenin temsili: son (rightmost) sembol → last-step
        symbols = [str(row[-1]) for row in sym_matrix]
        return symbols

    # ------------------------------------------------------------------
    # fit_transform kolaylık (YALNIZCA train için)
    # ------------------------------------------------------------------

    def fit_transform(self, train_signal: np.ndarray) -> list[str]:
        """fit() + transform() bileşimi. Test verisi için ÇAĞIRMAYIN."""
        self.fit(train_signal)
        return self.transform(train_signal)

    # ------------------------------------------------------------------
    # Leakage denetim
    # ------------------------------------------------------------------

    def assert_no_leakage(self, signal: np.ndarray) -> None:
        """
        transform(signal) çağrısının breakpoint'leri değiştirmediğini doğrular.

        Fırlatır
        --------
        AssertionError : Breakpoint'ler değişmişse (data leakage).
        RuntimeError   : fit() henüz çağrılmamışsa.
        """
        if not self._is_fitted:
            raise RuntimeError("Öncelikle fit() çağrılmalıdır.")
        before = self._breakpoints_snap.copy()
        _ = self.transform(signal)
        after = self._breakpoints.copy()
        if not np.allclose(before, after, equal_nan=True):
            raise AssertionError(
                f"DATA LEAKAGE TESPİT EDİLDİ!\n"
                f"  öncesi : {before}\n"
                f"  sonrası: {after}"
            )

    # ------------------------------------------------------------------
    # Breakpoint erişim
    # ------------------------------------------------------------------

    def _get_raw_breakpoints(self) -> np.ndarray:
        """Donmuş breakpoint dizisini döndürür."""
        if self._breakpoints is None:
            return np.array([])
        return self._breakpoints.copy()

    def get_breakpoints(self) -> np.ndarray | None:
        """Eğitilmiş SAX breakpoint değerlerini döndürür; fit edilmemişse None."""
        if not self._is_fitted:
            return None
        bps = self._get_raw_breakpoints()
        return bps if len(bps) > 0 else None

    def n_output_symbols(self, signal_length: int) -> int:
        """N uzunluklu sinyal için üretilecek SAX sembol sayısını döndürür."""
        return max(0, signal_length - self.paa_segments + 1)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"PAASAXTransformer("
            f"alphabet_size={self.alphabet_size}, "
            f"paa_segments={self.paa_segments}, "
            f"strategy={self.strategy}, "
            f"status={status})"
        )
