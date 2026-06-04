"""
explainer.py — Olasılıksal Otomata (PSA) karar açıklayıcısı.

Her karar için ``explain(window, time_step)`` fonksiyonu iki format üretir:

1. **JSON çıktı** (makine-okunur):
   {
     "time_step"      : int,          # pencere son adım indeksi
     "current_state"  : str,          # penceredeki son SAX pattern (son durum)
     "pattern"        : str,          # gözlemlenen SAX pattern dizisi (window)
     "status"         : "seen"|"unseen",
     "nearest_pattern": str | null,   # unseen → train'deki en yakın pattern
     "edit_distance"  : int | null,   # Levenshtein mesafesi (unseen ise)
     "transitions"    : [             # ardışık geçişler
         {"from": str, "to": str, "log_prob": float, "prob": float}
     ],
     "log_path_prob"  : float,        # Σ log(a_ij)  (ham toplam)
     "norm_log_prob"  : float,        # log_path / n_transitions
     "confidence"     : float,        # [0,1], yüksek = normal
     "interpretation" : str,          # "HIGH (normal)" / "LOW (anomaly)"
     "decision"       : "normal"|"anomaly"
   }

2. **İnsan-okunur tablo / metin** (konsol / rapor):
   Başlık + durum bilgisi + geçiş tablosu + özet satırı.

Deterministik garanti:
  Aynı ``window`` ve ``model`` girdisi → AYNI JSON ve tablo.
  Tüm sayılar model.trans_matrix_ / model._log_trans_matrix'ten doğrudan
  okunur; yuvarlama yalnızca görüntüleme katmanında yapılır.

Levenshtein implementasyonu:
  Dış bağımlılık yoktur; standart DP matrisi kullanılır.

Kullanım
--------
>>> from src.models.automata.automata import ProbabilisticAutomata
>>> from src.explain.explainer import AutomataExplainer
>>> exp = AutomataExplainer(model)
>>> result = exp.explain(sax_window_patterns, time_step=42)
>>> print(result.to_json())
>>> print(result.to_table())
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Levenshtein mesafesi (saf Python, no deps)
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """
    Klasik DP tabanlı Levenshtein (edit) mesafesi.

    Parametreler
    ------------
    s1, s2 : str
        Karşılaştırılacak iki string (SAX pattern).

    Döndürür
    --------
    int — Minimum düzenleme (ekleme/silme/değiştirme) sayısı.
    """
    m, n = len(s1), len(s2)
    # (m+1) × (n+1) DP matrisi
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # silme
                dp[i][j - 1] + 1,          # ekleme
                dp[i - 1][j - 1] + cost,   # değiştirme
            )
    return dp[m][n]


def _nearest_pattern(
    query: str, candidates: list[str]
) -> tuple[str, int]:
    """
    ``query`` string'ine Levenshtein mesafesi en küçük adayı döndürür.

    Parametreler
    ------------
    query      : str   — Aranan (unseen) pattern.
    candidates : list  — Aday pattern listesi (train state sözlüğü).

    Döndürür
    --------
    (nearest_pattern, distance) : (str, int)
    """
    best_pat, best_dist = candidates[0], int(1e9)
    for cand in candidates:
        d = _levenshtein(query, cand)
        if d < best_dist:
            best_dist = d
            best_pat = cand
    return best_pat, best_dist


# ---------------------------------------------------------------------------
# Sonuç veri yapısı
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    """
    Tek bir pencere kararının açıklama sonucu.

    Alanlar
    -------
    time_step       : int   — last-step indeksi (orijinal veri satırı)
    current_state   : str   — pencerenin son pattern'ı (aktif durum)
    pattern         : str   — pencerenin tüm SAX pattern'larını birleştiren temsil
    status          : str   — "seen" veya "unseen"
    nearest_pattern : str | None  — unseen ise en yakın train pattern'ı
    edit_distance   : int | None  — unseen ise Levenshtein mesafesi
    transitions     : list  — {"from", "to", "log_prob", "prob"} listesi
    log_path_prob   : float — Σ log(a_ij), ham toplam (log-space)
    norm_log_prob   : float — log_path / n_transitions (per-transition avg)
    confidence      : float — [0,1]; yüksek = normal
    interpretation  : str   — insan-okunur yorum
    decision        : str   — "normal" veya "anomaly"
    """
    time_step: int
    current_state: str
    pattern: str
    status: str
    nearest_pattern: str | None
    edit_distance: int | None
    transitions: list[dict[str, Any]]
    log_path_prob: float
    norm_log_prob: float
    confidence: float
    interpretation: str
    decision: str

    # ------------------------------------------------------------------
    # JSON çıktı
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """
        Sonucu JSON string'e dönüştürür.

        ``nan`` / ``inf`` değerleri string olarak korunur.

        Parametreler
        ------------
        indent : int, varsayılan 2
            JSON girintisi.

        Döndürür
        --------
        str — Geçerli JSON.
        """
        d = asdict(self)
        # float nan/inf → string (JSON geçerliliği için)
        def _clean(obj: Any) -> Any:
            if isinstance(obj, float):
                if math.isnan(obj):
                    return "NaN"
                if math.isinf(obj):
                    return "-Inf" if obj < 0 else "Inf"
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj
        return json.dumps(_clean(d), ensure_ascii=False, indent=indent)

    # ------------------------------------------------------------------
    # İnsan-okunur tablo
    # ------------------------------------------------------------------

    def to_table(self) -> str:
        """
        Sonucu insan-okunur tablo/metin formatında döndürür.

        Satırlar:
          ─ Başlık (zaman adımı, karar)
          ─ Durum bilgisi (pattern, status, nearest)
          ─ Geçiş tablosu (from | to | log-prob | prob)
          ─ Özet (path prob, confidence, interpretation)

        Döndürür
        --------
        str — Terminal'e basılabilir formatlı metin.
        """
        SEP = "-" * 68
        lines: list[str] = []

        # ── Başlık ──────────────────────────────────────────────────
        decision_icon = "⚠ ANOMALİ" if self.decision == "anomaly" else "✓ NORMAL"
        lines.append(SEP)
        lines.append(
            f"  KARAR AÇIKLAMASI  |  Zaman Adımı: {self.time_step}  |  {decision_icon}"
        )
        lines.append(SEP)

        # ── Durum Bilgisi ────────────────────────────────────────────
        lines.append(f"  Gözlemlenen Pattern : {self.pattern!r}")
        lines.append(f"  Aktif Durum (State) : {self.current_state!r}")
        lines.append(f"  Train'de Görülme    : {self.status.upper()}")
        if self.status == "unseen":
            lines.append(f"  En Yakın Pattern    : {self.nearest_pattern!r}")
            lines.append(f"  Levenshtein Mesafesi: {self.edit_distance}")
        lines.append("")

        # ── Geçiş Tablosu ───────────────────────────────────────────
        if self.transitions:
            col_w = [10, 10, 12, 10]
            header = (
                f"  {'Kaynak':^{col_w[0]}} {'Hedef':^{col_w[1]}} "
                f"{'log-Prob':^{col_w[2]}} {'Olasılık':^{col_w[3]}}"
            )
            lines.append(header)
            lines.append("  " + "·" * (sum(col_w) + 3))
            for t in self.transitions:
                log_p = t["log_prob"]
                prob  = t["prob"]
                lines.append(
                    f"  {t['from']:^{col_w[0]}} "
                    f"{t['to']:^{col_w[1]}} "
                    f"{log_p:^{col_w[2]}.6f} "
                    f"{prob:^{col_w[3]}.6f}"
                )
        else:
            lines.append("  (Geçiş bulunamadı — pencere çok kısa)")
        lines.append("")

        # ── Özet ────────────────────────────────────────────────────
        lines.append(f"  Ham Log-Path Prob   : {self.log_path_prob:.6f}")
        lines.append(f"  Norm. Log-Path Prob : {self.norm_log_prob:.6f}")
        lines.append(f"  Güven Skoru         : {self.confidence:.4f}  ({self.interpretation})")
        lines.append(f"  Nihai Karar         : {self.decision.upper()}")
        lines.append(SEP)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ana açıklayıcı sınıf
# ---------------------------------------------------------------------------


class AutomataExplainer:
    """
    ``ProbabilisticAutomata`` modelinin kararlarını açıklar.

    Parametreler
    ------------
    model : ProbabilisticAutomata
        Fit edilmiş PSA modeli. ``model.is_fitted_`` True olmalıdır.

    Örnek
    -----
    >>> exp = AutomataExplainer(model)
    >>> result = exp.explain(["ab", "ba", "ac"], time_step=10)
    >>> print(result.to_json())
    >>> print(result.to_table())
    """

    # Güven skoru yorumlama eşiği (config-driven)

    def __init__(self, model: Any, config: dict | None = None) -> None:
        if not getattr(model, "is_fitted_", False):
            raise RuntimeError(
                "AutomataExplainer: model henüz fit edilmemiş. Önce model.fit() çağrın."
            )
        self._model = model
        # confidence_high_threshold config'den oku, yoksa varsayılan 0.5
        if config is not None and "explain" in config:
            self._conf_high_thresh = float(
                config["explain"].get("confidence_high_threshold", 0.5)
            )
        else:
            self._conf_high_thresh = 0.5

    # ------------------------------------------------------------------
    # Yardımcı: tek geçişin log/doğrusal olasılığı
    # ------------------------------------------------------------------

    def _transition_prob(self, src: str, dst: str) -> tuple[float, float]:
        """
        (src → dst) geçişinin (log_prob, linear_prob) değerlerini döndürür.

        Bilinmeyen state durumunda uniform log-prob kullanılır (model._compute_path_scores
        ile aynı davranış — deterministik tutarlılık).

        Döndürür
        --------
        (log_prob, linear_prob) : (float, float)
        """
        M = self._model.n_states
        log_unseen = math.log(1.0 / M) if M > 0 else float("-inf")

        src_idx = self._model.state_index_.get(src)
        dst_idx = self._model.state_index_.get(dst)

        if src_idx is None or dst_idx is None:
            lp = log_unseen
        else:
            lp = float(self._model._log_trans_matrix[src_idx, dst_idx])

        return lp, math.exp(lp)

    # ------------------------------------------------------------------
    # Yardımcı: güven skoru → yorum
    # ------------------------------------------------------------------

    def _interpret_confidence(self, conf: float) -> str:
        """
        Güven skorunu "HIGH (normal)" ya da "LOW (anomaly)" olarak yorumlar.
        """
        if conf >= self._conf_high_thresh:
            return "HIGH (normal)"
        return "LOW (anomaly)"

    # ------------------------------------------------------------------
    # Ana açıklama fonksiyonu
    # ------------------------------------------------------------------

    def explain(
        self,
        window_patterns: list[str],
        time_step: int,
    ) -> ExplanationResult:
        """
        Tek bir pencere için otomata kararını açıklar.

        Parametreler
        ------------
        window_patterns : list[str]
            Sliding window ile üretilmiş ardışık SAX pattern listesi.
            Örn. ``["ab", "ba", "ac"]`` (window_size=2 ile).
            En az 1 eleman, anlamlı geçiş için en az 2 eleman gereklidir.
        time_step : int
            Pencerenin son satırının orijinal veri indeksi (last-step kuralı).

        Döndürür
        --------
        ExplanationResult
            JSON ve tablo formatında sunulabilen açıklama nesnesi.

        Notlar
        ------
        - Tüm olasılık değerleri ``model.trans_matrix_`` / ``model._log_trans_matrix``
          'ten doğrudan okunur → deterministik ve matrisle birebir tutarlı.
        - ``window_patterns`` boşsa veya tek elemanlıysa geçiş listesi boş,
          log_path_prob ``nan`` olur; karar eşik karşılaştırması yapılamadığından
          ``"normal"`` döner.
        """
        model = self._model
        known_patterns: list[str] = list(model.state_index_.keys())

        # ── Aktif durum = penceredeki son pattern ────────────────────
        current_state = window_patterns[-1] if window_patterns else ""

        # ── Pattern temsili (birleştirme) ────────────────────────────
        pattern_repr = " → ".join(window_patterns) if window_patterns else "(boş)"

        # ── Seen / Unseen kontrolü ───────────────────────────────────
        is_seen = current_state in model.state_index_
        status = "seen" if is_seen else "unseen"
        nearest_pat: str | None = None
        edit_dist: int | None = None
        if not is_seen and known_patterns:
            nearest_pat, edit_dist = _nearest_pattern(current_state, known_patterns)

        # ── Geçiş tablosu ────────────────────────────────────────────
        transitions: list[dict[str, Any]] = []
        log_probs_collected: list[float] = []

        for i in range(len(window_patterns) - 1):
            src = window_patterns[i]
            dst = window_patterns[i + 1]
            lp, p = self._transition_prob(src, dst)
            transitions.append(
                {
                    "from"    : src,
                    "to"      : dst,
                    "log_prob": lp,
                    "prob"    : p,
                }
            )
            log_probs_collected.append(lp)

        # ── Path probability (log ve normalize) ──────────────────────
        if log_probs_collected:
            log_path_prob = float(np.sum(log_probs_collected))
            norm_log_prob = log_path_prob / len(log_probs_collected)
        else:
            log_path_prob = float("nan")
            norm_log_prob = float("nan")

        # ── Güven skoru ──────────────────────────────────────────────
        if not math.isnan(norm_log_prob):
            confidence = model.confidence_score(norm_log_prob)
        else:
            confidence = float("nan")

        interpretation = (
            self._interpret_confidence(confidence)
            if not math.isnan(confidence)
            else "BILINMIYOR"
        )

        # ── Karar ────────────────────────────────────────────────────
        if math.isnan(norm_log_prob):
            decision = "normal"          # geçiş yoksa karar verilemez → varsayılan normal
        else:
            decision = (
                "anomaly"
                if norm_log_prob < float(model.threshold_)
                else "normal"
            )

        return ExplanationResult(
            time_step=time_step,
            current_state=current_state,
            pattern=pattern_repr,
            status=status,
            nearest_pattern=nearest_pat,
            edit_distance=edit_dist,
            transitions=transitions,
            log_path_prob=log_path_prob,
            norm_log_prob=norm_log_prob,
            confidence=confidence,
            interpretation=interpretation,
            decision=decision,
        )

    # ------------------------------------------------------------------
    # Toplu açıklama (opsiyonel kolaylık)
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        all_patterns: list[list[str]],
        base_time_step: int = 0,
    ) -> list[ExplanationResult]:
        """
        Birden fazla pencere için toplu açıklama üretir.

        Parametreler
        ------------
        all_patterns : list[list[str]]
            Her eleman bir ``window_patterns`` listesi.
        base_time_step : int
            İlk pencerenin time_step değeri (sonrakiler +1 artar).

        Döndürür
        --------
        list[ExplanationResult]
        """
        return [
            self.explain(wp, time_step=base_time_step + i)
            for i, wp in enumerate(all_patterns)
        ]


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

def _demo():
    """
    Küçük sahte model üzerinde 1 SEEN + 1 UNSEEN pencere açıklaması.

    Bu fonksiyon harici bağımlılık gerekmeden çalışır:
      - ProbabilisticAutomata sınıfını import eder,
      - minimal bir train sembol dizisiyle fit eder,
      - iki farklı pencere için explain() çağırır,
      - JSON + tablo çıktıları ekrana basar.
    """
    import sys, os
    # Windows konsolunda Turkce / Unicode karakterlerin dogru gozukmesi icin
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Proje kök dizinini ekle (doğrudan çalıştırma için)
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)


    from src.models.automata.automata import ProbabilisticAutomata

    # ── Sahte config ─────────────────────────────────────────────────
    config = {
        "automata": {
            "window_size"       : 2,
            "smoothing_k"       : 0.1,
            "anomaly_percentile": 5,
        }
    }

    # ── Sahte train sembolleri ────────────────────────────────────────
    # Alfabe: {a, b, c}; tekrarlı dizi → bazı geçişler sık, bazıları nadir
    train_symbols = list("aababcbabcabababcbab")
    model = ProbabilisticAutomata(config)
    model.fit(train_symbols)

    print("=" * 68)
    print("  ProbabilisticAutomata - fit ozeti")
    print("=" * 68)
    print(f"  Model     : {model}")
    print(f"  State sayı: {model.n_states}")
    print(f"  Eşik      : {model.threshold_:.6f}")
    print(f"  States    : {list(model.state_index_.keys())}")
    print()

    exp = AutomataExplainer(model)

    # ── ÖRNEK 1: SEEN pencere ────────────────────────────────────────
    # window_size=2 → 2-gram pattern'lar; her ikisi de train'de görülen
    seen_window = ["ab", "ba"]     # her iki pattern train'de mevcut
    result_seen = exp.explain(seen_window, time_step=5)

    print("=" * 68)
    print("  ORNEK 1 - SEEN Pencere")
    print("=" * 68)
    print(result_seen.to_json())
    print()
    print(result_seen.to_table())
    print()

    # ── ÖRNEK 2: UNSEEN pencere ──────────────────────────────────────
    # "xz" pattern'ı train'de hiç geçmemiş → unseen; nearest hesaplanır
    unseen_window = ["ab", "xz"]   # "xz" kesinlikle unseen
    result_unseen = exp.explain(unseen_window, time_step=12)

    print("=" * 68)
    print("  ORNEK 2 - UNSEEN Pencere")
    print("=" * 68)
    print(result_unseen.to_json())
    print()
    print(result_unseen.to_table())


if __name__ == "__main__":
    _demo()
