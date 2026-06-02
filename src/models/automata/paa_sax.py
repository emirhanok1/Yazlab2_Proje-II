"""
paa_sax.py — pyts PAA + SAX (quantile strategy) islemi.

Mimari:
  1. Sinyal sliding window ile (n_windows, paa_segments) matrisine donusturulur.
     Her satir = paa_segments zamanli bir pencere.
  2. pyts PiecewiseAggregateApproximation:
       (n_windows, paa_segments) -> (n_windows, paa_out) [paa_out <= paa_segments]
     Burada paa_out = n_sax_bins (sembol sayisi, varsayilan paa_segments).
  3. pyts SymbolicAggregateApproximation(strategy='quantile'):
       fit()       -> train pencerelerinden quantile breakpoint'leri ogren -> DONDUR
       transform() -> donmus breakpoint'lerle sembolize et

  Son cikti: (n_windows,) uzunlugunda string SAX sembol listesi.
  Her eleman tek bir SAX karakteridir ('a', 'b', 'c', ...).
  n_windows = N - paa_segments + 1.

Leakage garantisi:
  - _sax.fit() YALNIZCA PAASAXTransformer.fit() icinde cagrilir.
  - PAASAXTransformer.transform() icinde SADECE _sax.transform() cagrilir.
  - fit() oncesi transform() cagrilirsa RuntimeError firlatin.

Parametreler (config['automata']):
  alphabet_size : SAX alfabe buyuklugu (n_bins)
  paa_segments  : Sliding window boyutu = PAA timestamp sayisi
"""

from __future__ import annotations

import numpy as np
from pyts.approximation import SymbolicAggregateApproximation


class PAASAXTransformer:
    """
    Sliding-window + pyts SAX (quantile) donusumu; sklearn-benzeri fit/transform.

    Parametreler
    ------------
    config : dict
        config.yaml icerigi; config['automata']['alphabet_size'] ve
        config['automata']['paa_segments'] anahtarlari zorunludur.

    Dahili durum
    ------------
    _sax                : SymbolicAggregateApproximation (fit sonrasi dolu)
    _breakpoints_snap   : np.ndarray  (fit anindaki breakpoint kopyas)
    _is_fitted          : bool
    """

    def __init__(self, config: dict):
        auto = config["automata"]
        self.alphabet_size: int = int(auto["alphabet_size"])
        self.paa_segments: int  = int(auto["paa_segments"])
        self._is_fitted: bool = False
        self._sax: SymbolicAggregateApproximation | None = None
        self._breakpoints_snap: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Dahili: sliding-window matris olustur
    # ------------------------------------------------------------------

    def _make_windows(self, signal: np.ndarray) -> np.ndarray:
        """
        1D sinyali (N,) sliding-window matrisi (n_windows, paa_segments)'e cevirir.

        n_windows = N - paa_segments + 1.
        Her satir W = paa_segments zamanli ardisik pencere.
        """
        sig = np.asarray(signal, dtype=float).ravel()
        N, W = len(sig), self.paa_segments
        if N < W:
            raise ValueError(
                f"Sinyal uzunlugu ({N}) paa_segments'ten ({W}) kucuk olamaz."
            )
        # np.lib.stride_tricks.sliding_window_view -- NumPy >= 1.20
        windows = np.lib.stride_tricks.sliding_window_view(sig, W).copy()
        # shape: (N - W + 1, W)
        return windows

    # ------------------------------------------------------------------
    # fit -- YALNIZCA train
    # ------------------------------------------------------------------

    def fit(self, train_signal: np.ndarray) -> "PAASAXTransformer":
        """
        SAX breakpoint'lerini yalnizca train sinyalinin sliding penceleri uzerinde ogrenir.

        Parametreler
        ------------
        train_signal : array-like, shape (N,)
            Train PC1 sinyali. N >= paa_segments olmali.

        Adimlar
        -------
        1. Train sinyal -> sliding-window matrisi (n_train_wins, paa_segments).
        2. SAX.fit(matris): her sutunun quantile'larindan breakpoint'ler ogren.
           - strategy='quantile' -> veri dagilimina dayali; DONDUR.
        3. Breakpoint anlık kopya sakla (leakage testi icin).

        Donus
        -----
        self
        """
        windows = self._make_windows(train_signal)  # (n_wins, W)

        # pyts SAX: (n_samples, n_timestamps) -> her timestamp -> breakpoint
        # n_bins=alphabet_size, strategy='quantile'
        self._sax = SymbolicAggregateApproximation(
            n_bins=self.alphabet_size,
            strategy="quantile",
            alphabet=None,          # varsayilan: 'a','b','c',...
        )
        self._sax.fit(windows)      # breakpoint'leri SADECE buradan ogren

        # Breakpoint anlık kopya -- leakage denetimi icin
        self._breakpoints_snap = self._get_raw_breakpoints().copy()
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform -- donmus breakpoint'lerle herhangi veriye uygula
    # ------------------------------------------------------------------

    def transform(self, signal: np.ndarray) -> list[str]:
        """
        Donmus breakpoint'lerle sinyali SAX sembol listesine donusturur.

        Parametreler
        ------------
        signal : array-like, shape (N,)
            Donusturulecek sinyal (train/val/test).

        Donus
        -----
        symbols : list[str]
            Uzunluk = N - paa_segments + 1.
            Her eleman tek karakterlik SAX sembol ('a', 'b', 'c', ...).

        Leakage notu
        ------------
        Bu metot icinde self._sax.fit() CAGRILMAZ.
        Yalnizca .transform() kullanilir -> breakpoint'ler degismez.
        """
        if not self._is_fitted:
            raise RuntimeError("transform() oncesi fit() cagrilmalidir.")

        windows = self._make_windows(signal)  # (n_wins, W)
        # SAX transform: donmus breakpoint'lerle sembolize et
        # cikti shape: (n_wins, W), dtype: str (tek karakter)
        X_sax = self._sax.transform(windows)  # (n_wins, W)

        # Her pencerenin temsili: son (rightmost) sembol
        # -> last-step hizalamasiyla tutarli (DL ile simetri)
        symbols = [str(row[-1]) for row in X_sax]
        return symbols

    # ------------------------------------------------------------------
    # fit_transform kolaylik (YALNIZCA train icin)
    # ------------------------------------------------------------------

    def fit_transform(self, train_signal: np.ndarray) -> list[str]:
        """fit() + transform() bilesimi. Test verisi icin CAGIRMAYIN."""
        self.fit(train_signal)
        return self.transform(train_signal)

    # ------------------------------------------------------------------
    # Leakage denetim
    # ------------------------------------------------------------------

    def assert_no_leakage(self, signal: np.ndarray) -> None:
        """
        transform(signal) cagrisinin breakpoint'leri degistirmedigini dogrular.

        Firlatir
        --------
        AssertionError : Breakpoint'ler degismisse (data leakage).
        RuntimeError   : fit() henuz cagrilmamissa.
        """
        if not self._is_fitted:
            raise RuntimeError("Oncelikle fit() cagrilmalidir.")
        before = self._breakpoints_snap.copy()
        _ = self.transform(signal)
        after = self._get_raw_breakpoints()
        if len(before) > 0 and len(after) > 0:
            if not np.allclose(before, after, equal_nan=True):
                raise AssertionError(
                    f"DATA LEAKAGE TESPIT EDILDI!\n"
                    f"  oncesi : {before}\n"
                    f"  sonrasi: {after}"
                )

    # ------------------------------------------------------------------
    # Dahili: breakpoint cekme
    # ------------------------------------------------------------------

    def _get_raw_breakpoints(self) -> np.ndarray:
        """pyts SAX'tan breakpoint degerlerini ceker (pyts version-safe)."""
        if self._sax is None:
            return np.array([])
        # pyts >= 0.12 -> bin_edges_ attribute (shape: n_bins-1 veya 2D)
        if hasattr(self._sax, "bin_edges_"):
            return np.asarray(self._sax.bin_edges_).ravel()
        # Fallback: breakpoints_ attribute (eski pyts)
        if hasattr(self._sax, "breakpoints_"):
            return np.asarray(self._sax.breakpoints_).ravel()
        return np.array([])

    def get_breakpoints(self) -> np.ndarray | None:
        """Egitilmis SAX breakpoint degerlerini dondurur; fit edilmemisse None."""
        if not self._is_fitted:
            return None
        bps = self._get_raw_breakpoints()
        return bps if len(bps) > 0 else None

    def n_output_symbols(self, signal_length: int) -> int:
        """N uzunluklu sinyal icin uretilecek SAX sembol sayisini dondurur."""
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
            f"status={status})"
        )
