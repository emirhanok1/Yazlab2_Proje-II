"""
unseen.py — Levenshtein tabanli unseen pattern yonetimi.

Giris: ProbabilisticAutomata.state_index_ sozlugu (train SAX pattern'lari).
Cikis: Test pattern'lari icin seen/unseen siniflandirmasi + nearest-neighbor mapping.

Isleyis:
  1. UnseenHandler(state_index) ile train pattern sozlugunu kaydet.
  2. lookup(pattern) -> bilinen pattern ise dogrudan donus; unseen ise
     Levenshtein edit distance ile en yakin train pattern'i bul.
  3. score_sequence(patterns) -> her pattern icin (resolved_pattern, distance, is_unseen).
  4. detection_rate() -> unseen oran.
  5. mapping_report() -> unseen -> nearest eslesme ozeti.

Levenshtein:
  - python-Levenshtein kutuphanesi (Levenshtein.distance).
  - Fallback: elle dinamik programlama (DP) implementasyonu.
    requirements.txt'te python-Levenshtein zaten var.

Kural:
  - Unseen pattern tamamen bilinmeyen; state_index sozlugunde YOK.
  - Eslenen (resolved) pattern state_index'te VARDIR -> otomata devam eder.
  - Esit uzaklikta birden fazla varsa: ilk alfabetik olan secilir (deterministik).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import Levenshtein as _lev_lib          # python-Levenshtein
    def _edit_distance(a: str, b: str) -> int:
        return _lev_lib.distance(a, b)
    _LEV_SOURCE = "python-Levenshtein"
except ImportError:                         # Fallback: elle DP
    def _edit_distance(a: str, b: str) -> int:  # type: ignore[misc]
        """Wagner-Fischer dinamik programlama ile Levenshtein mesafesi."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]
    _LEV_SOURCE = "builtin-DP"


# ---------------------------------------------------------------------------
# Veri sinifi: tek bir pattern icin arama sonucu
# ---------------------------------------------------------------------------

@dataclass
class LookupResult:
    """
    Bir pattern arama sonucunu temsil eder.

    Alanlar
    -------
    query      : str   — sorgu pattern
    resolved   : str   — eslenen (bilinen) pattern; seen ise query ile ayni
    distance   : int   — edit distance (seen icin 0)
    is_unseen  : bool  — True ise train sozlugunde yoktu
    state_id   : int   — resolved pattern'in state indeksi
    """
    query:     str
    resolved:  str
    distance:  int
    is_unseen: bool
    state_id:  int

    def __repr__(self) -> str:
        tag = "UNSEEN" if self.is_unseen else "SEEN"
        return (
            f"LookupResult({tag} query={self.query!r} -> "
            f"resolved={self.resolved!r} dist={self.distance} "
            f"state={self.state_id})"
        )


# ---------------------------------------------------------------------------
# Ana sinif
# ---------------------------------------------------------------------------

class UnseenHandler:
    """
    Train SAX pattern sozlugune dayali unseen pattern yoneticisi.

    Kullanim
    --------
    >>> handler = UnseenHandler(model.state_index_)
    >>> result  = handler.lookup("adc")
    >>> results = handler.score_sequence(test_patterns)
    >>> print(handler.detection_rate())
    >>> print(handler.mapping_report())

    Parametreler
    ------------
    state_index : dict[str, int]
        ProbabilisticAutomata.state_index_ sozlugu.
        pattern -> 0-tabanli state indeksi.

    Dahili durum (session bazli, reset ile sifirlanir)
    ---------------------------------------------------
    _lookup_log : list[LookupResult]
        score_sequence() veya lookup() sonrasi biriken arama kaydi.
    """

    def __init__(self, state_index: dict[str, int]):
        if not state_index:
            raise ValueError("state_index bos olamaz — fit() sonrasi kullanin.")
        self._state_index: dict[str, int] = dict(state_index)
        # Deterministik siralama: alfabetik liste
        self._known_patterns: list[str] = sorted(self._state_index.keys())
        self._lookup_log: list[LookupResult] = []

    # ------------------------------------------------------------------
    # Ana arama metodu
    # ------------------------------------------------------------------

    def lookup(self, pattern: str) -> LookupResult:
        """
        Tek bir pattern icin seen/unseen siniflandirmasi + nearest-neighbor bulma.

        Parametreler
        ------------
        pattern : str
            Sorgu SAX pattern string'i (ornk. 'adc', 'aabb').

        Donus
        -----
        LookupResult
            - Seen ise: resolved=pattern, distance=0, is_unseen=False.
            - Unseen ise: resolved=nearest, distance>=1, is_unseen=True.

        Notlar
        ------
        - Esit uzaklikta birden fazla aday varsa: alfabetik kucuk olan secilir.
        - Sonuc _lookup_log'a eklenir (istatistik icin).
        """
        if pattern in self._state_index:
            result = LookupResult(
                query=pattern,
                resolved=pattern,
                distance=0,
                is_unseen=False,
                state_id=self._state_index[pattern],
            )
        else:
            nearest, dist = self._nearest_neighbor(pattern)
            result = LookupResult(
                query=pattern,
                resolved=nearest,
                distance=dist,
                is_unseen=True,
                state_id=self._state_index[nearest],
            )

        self._lookup_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Toplu dizi skorlama
    # ------------------------------------------------------------------

    def score_sequence(self, patterns: list[str]) -> list[LookupResult]:
        """
        Pattern listesi icin toplu lookup; sonuclari dondurur ve loglar.

        Parametreler
        ------------
        patterns : list[str]
            Otomata'nin sliding window'dan urettigi pattern listesi.

        Donus
        -----
        list[LookupResult]
            Her pattern icin ayri LookupResult.
        """
        return [self.lookup(p) for p in patterns]

    # ------------------------------------------------------------------
    # Istatistik raporlama
    # ------------------------------------------------------------------

    def detection_rate(self) -> float:
        """
        Son score_sequence/lookup oturumundaki unseen oranini dondurur.

        Donus
        -----
        float : [0.0, 1.0] — unseen_sayisi / toplam_sorgu.
        0 sorgu yapilmissa 0.0 doner.
        """
        if not self._lookup_log:
            return 0.0
        n_unseen = sum(1 for r in self._lookup_log if r.is_unseen)
        return n_unseen / len(self._lookup_log)

    def mapping_report(self) -> dict:
        """
        Unseen -> nearest mapping ozetini sozluk olarak dondurur.

        Donus
        -----
        dict anahtarlari:
          total_queries    : int   — toplam sorgu sayisi
          n_seen           : int   — seen sayisi
          n_unseen         : int   — unseen sayisi
          detection_rate   : float — unseen orani
          avg_distance     : float — unseen icin ortalama edit distance
          max_distance     : int   — unseen icin maksimum edit distance
          mappings         : list[dict]  — her unseen icin detay
        """
        total = len(self._lookup_log)
        unseen_results = [r for r in self._lookup_log if r.is_unseen]
        n_unseen = len(unseen_results)
        n_seen = total - n_unseen

        avg_dist = (
            sum(r.distance for r in unseen_results) / n_unseen
            if n_unseen > 0 else 0.0
        )
        max_dist = (
            max(r.distance for r in unseen_results)
            if n_unseen > 0 else 0
        )

        mappings = [
            {
                "query": r.query,
                "resolved": r.resolved,
                "distance": r.distance,
                "state_id": r.state_id,
            }
            for r in unseen_results
        ]

        return {
            "total_queries":  total,
            "n_seen":         n_seen,
            "n_unseen":       n_unseen,
            "detection_rate": self.detection_rate(),
            "avg_distance":   round(avg_dist, 4),
            "max_distance":   max_dist,
            "mappings":       mappings,
        }

    def reset_log(self) -> None:
        """Lookup log'unu temizler (yeni oturum icin)."""
        self._lookup_log = []

    # ------------------------------------------------------------------
    # Yardimci: nearest-neighbor
    # ------------------------------------------------------------------

    def _nearest_neighbor(self, query: str) -> tuple[str, int]:
        """
        Train sozlugunde query'ye en kucuk Levenshtein mesafeli pattern'i bulur.

        Esit mesafede birden fazla aday varsa: alfabetik kucuk olan (deterministik).

        Parametreler
        ------------
        query : str — sozlukte OLMAYAN sorgu pattern.

        Donus
        -----
        (nearest_pattern, min_distance) : tuple[str, int]
        """
        best_pattern: str = self._known_patterns[0]
        best_dist: int = _edit_distance(query, best_pattern)

        for candidate in self._known_patterns[1:]:
            d = _edit_distance(query, candidate)
            if d < best_dist or (d == best_dist and candidate < best_pattern):
                best_dist = d
                best_pattern = candidate

        return best_pattern, best_dist

    # ------------------------------------------------------------------
    # Bilgi sorgulama
    # ------------------------------------------------------------------

    @property
    def n_known_patterns(self) -> int:
        """Train sozlugundeki benzersiz pattern sayisi."""
        return len(self._state_index)

    @property
    def known_patterns(self) -> list[str]:
        """Alfabetik sirali bilinen pattern listesi (kopya)."""
        return list(self._known_patterns)

    @property
    def levenshtein_source(self) -> str:
        """Kullanilan Levenshtein implementasyonu."""
        return _LEV_SOURCE

    def is_seen(self, pattern: str) -> bool:
        """Pattern train sozlugunde var mi?"""
        return pattern in self._state_index

    def __repr__(self) -> str:
        return (
            f"UnseenHandler("
            f"n_known={self.n_known_patterns}, "
            f"lev_source={_LEV_SOURCE!r}, "
            f"log_size={len(self._lookup_log)})"
        )


# ---------------------------------------------------------------------------
# Modul duzeyinde yardimci: edit_distance (dis erisim icin)
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """
    Iki string arasindaki Levenshtein (edit) mesafesini hesaplar.

    python-Levenshtein yuklu ise onu kullanir, yoksa yerlesik DP'ye duser.

    Parametreler
    ------------
    a, b : str — karsilastirilacak stringler.

    Donus
    -----
    int — minimum duzenle mesafesi (>= 0).
    """
    return _edit_distance(a, b)
