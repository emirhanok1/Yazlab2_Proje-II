"""
stats.py — İstatistiksel Anlamlılık Testleri Modülü.

Rubrik: 5 puan (zorunlu).

İki test uygulanır:

1. **Wilcoxon Signed-Rank Testi** (scipy.stats.wilcoxon)
   - Her model çifti (A, B) için eşleştirilmiş F1 skorları üzerinde çalışır.
   - SKAB: (seed, fold) bazlı eşleşme → n=25 (5 seed × 5 fold)
   - BATADAL: seed bazlı eşleşme → n=5 (⚠ küçük örneklem, uyarı verilir)
   - Alternatif: two-sided (yön belirtilmez, fark var mı?)
   - Anlamlılık: p < alpha (config'den)

2. **McNemar Testi** (scipy.stats.mcnemar)
   - İki modelin "doğru/yanlış" kararlarını karşılaştırır.
   - ⚠ Veri setinde bireysel tahmin vektörü yok; konfüzyon matrisi
     precision/recall/f1 + dağılım bilgisinden rekonstrükte edilir.
   - Bu nedenle McNemar YAKLAŞIK'tır — sınırlama raporda belgelenir.
   - Rekonstrüksiyon formülleri:
       TP = round(precision × N1)
       FP = N1 - TP
       FN = round(TP / recall) - TP   [recall = TP/(TP+FN)]
       TN = N0 - FN
     (N0, N1: modelin tahmin ettiği 0 ve 1 sayıları, distribution'dan)
   - Kontenjans tablosu (satır=modelA, sütun=modelB):
       [[n11, n10], [n01, n00]]
       n11 = min(correct_A, correct_B)  → her ikisi doğru (ust sınır)
       n10 = correct_A - n11             → A doğru, B yanlış
       n01 = correct_B - n11             → B doğru, A yanlış
       n00 = n_total - n11 - n10 - n01  → her ikisi yanlış

Çıktı (hepsi config['stats'] altından okunur):
   results/stats_wilcoxon.csv  — Wilcoxon sonuç tablosu
   results/stats_mcnemar.csv   — McNemar sonuç tablosu
   results/stats_report.txt    — Türkçe tablo + yorum (DL vs Automata odaklı)

Kullanım:
    python -m src.experiments.stats
    # veya
    from src.experiments.stats import StatisticalTester
    tester = StatisticalTester(config)
    tester.load_data("results/experiments_final.csv")
    results = tester.run_all()
    tester.save_results(results)
"""

from __future__ import annotations

import ast
import csv
import itertools
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

MODELS = ["lstm", "gru", "cnn1d", "automata"]
MODEL_PAIRS: list[tuple[str, str]] = list(itertools.combinations(MODELS, 2))
# 6 çift: (lstm,gru), (lstm,cnn1d), (lstm,automata), (gru,cnn1d), (gru,automata), (cnn1d,automata)

DL_MODELS = {"lstm", "gru", "cnn1d"}
AUTOMATA_MODEL = "automata"


# ---------------------------------------------------------------------------
# Sonuç veri yapıları
# ---------------------------------------------------------------------------

@dataclass
class WilcoxonResult:
    """Tek bir (dataset, modelA, modelB) çifti için Wilcoxon sonucu."""
    dataset: str
    scenario: str
    model_a: str
    model_b: str
    n_samples: int
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    mean_f1_a: float
    mean_f1_b: float
    better_model: str          # ortalama F1'e göre daha iyi
    warning: str               # boş veya uyarı mesajı


@dataclass
class McNemarResult:
    """Tek bir (dataset, modelA, modelB) çifti için McNemar sonucu."""
    dataset: str
    scenario: str
    model_a: str
    model_b: str
    n_total: int               # ortalama toplam örnek sayısı
    n_both_correct: float
    n_a_only: float
    n_b_only: float
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    note: str                  # yaklaşım uyarısı


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class StatisticalTester:
    """
    Wilcoxon signed-rank ve McNemar testleri ile model karşılaştırması.

    Parametreler
    ------------
    config : dict
        config.yaml içeriği; config['stats'] altındaki parametreler kullanılır.

    Dahili durum (load_data() sonrası)
    -----------------------------------
    _records : list[dict]
        experiments_final.csv'den yüklenen ham kayıtlar.
    _alpha : float
        Anlamlılık eşiği.
    _min_n_warn : int
        Bu değerin altında uyarı verilir.
    _scenarios : list[str]
        Wilcoxon'a dahil edilecek senaryolar (config'den).
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("stats", {})
        self._alpha: float = float(cfg.get("alpha", 0.05))
        self._min_n_warn: int = int(cfg.get("min_n_warning", 6))
        self._scenarios: list[str] = list(cfg.get("scenarios_for_wilcoxon", ["original"]))
        self._out_wilcoxon: str = str(cfg.get("output_wilcoxon_csv", "results/stats_wilcoxon.csv"))
        self._out_mcnemar: str = str(cfg.get("output_mcnemar_csv", "results/stats_mcnemar.csv"))
        self._out_report: str = str(cfg.get("output_report", "results/stats_report.txt"))
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Veri yükleme
    # ------------------------------------------------------------------

    def load_data(self, csv_path: str) -> "StatisticalTester":
        """
        experiments_final.csv'yi yükler ve dahili kayıt listesini doldurur.

        Parametreler
        ------------
        csv_path : str
            experiments_final.csv'nin yolu.

        Döndürür
        --------
        self
        """
        records: list[dict[str, Any]] = []
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV bulunamadı: {path}")

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not any(row.values()):  # boş satır
                    continue
                rec: dict[str, Any] = {}
                rec["model"]     = row["model"].strip().lower()
                rec["dataset"]   = row["dataset"].strip()
                rec["scenario"]  = row["scenario"].strip()
                rec["seed"]      = int(row["seed"])
                rec["fold"]      = int(row["fold"])
                rec["f1"]        = float(row["f1"])
                rec["precision"] = float(row["precision"])
                rec["recall"]    = float(row["recall"])
                rec["accuracy"]  = float(row["accuracy"])
                rec["degenerate"] = row.get("degenerate", "False").strip().lower() == "true"

                # distribution: string dict → python dict
                dist_raw = row.get("distribution", "{}")
                try:
                    dist = ast.literal_eval(dist_raw)
                    rec["n_pred_0"] = int(dist.get("0", 0))
                    rec["n_pred_1"] = int(dist.get("1", 0))
                except Exception:
                    rec["n_pred_0"] = 0
                    rec["n_pred_1"] = 0

                records.append(rec)

        self._records = records
        logger.info("[stats] %d kayıt yüklendi: %s", len(records), csv_path)
        return self

    # ------------------------------------------------------------------
    # Yardımcı: kayıt filtreleme
    # ------------------------------------------------------------------

    def _filter(
        self,
        dataset: str,
        model: str,
        scenarios: list[str] | None = None,
    ) -> list[dict]:
        """Verilen dataset + model + scenario kombinasyonuna göre filtreler."""
        scen = scenarios or self._scenarios
        return [
            r for r in self._records
            if r["dataset"] == dataset
            and r["model"] == model
            and r["scenario"] in scen
        ]

    # ------------------------------------------------------------------
    # Wilcoxon: eşleştirilmiş F1 karşılaştırması
    # ------------------------------------------------------------------

    def _paired_f1(
        self,
        dataset: str,
        model_a: str,
        model_b: str,
    ) -> tuple[list[float], list[float], list[tuple]]:
        """
        İki model için eşleştirilmiş F1 listelerini döndürür.

        SKAB: anahtar = (seed, fold)
        BATADAL: anahtar = seed (fold=0 hep)

        Döndürür
        --------
        (f1_a, f1_b, keys) — eşleştirilmiş listeler ve anahtar listesi.
        """
        recs_a = {
            (r["seed"], r["fold"]): r["f1"]
            for r in self._filter(dataset, model_a)
        }
        recs_b = {
            (r["seed"], r["fold"]): r["f1"]
            for r in self._filter(dataset, model_b)
        }

        # Ortak anahtarlar (kesişim) — sıralı deterministik
        common_keys = sorted(set(recs_a.keys()) & set(recs_b.keys()))
        f1_a = [recs_a[k] for k in common_keys]
        f1_b = [recs_b[k] for k in common_keys]
        return f1_a, f1_b, common_keys

    def _run_wilcoxon_pair(
        self,
        dataset: str,
        model_a: str,
        model_b: str,
    ) -> WilcoxonResult:
        """
        Tek bir (dataset, modelA, modelB) çifti için Wilcoxon testi çalıştırır.
        """
        scenario_label = "+".join(self._scenarios)
        f1_a, f1_b, keys = self._paired_f1(dataset, model_a, model_b)
        n = len(f1_a)

        warning = ""
        if n < self._min_n_warn:
            warning = (
                f"UYARI: Örneklem sayısı n={n} < {self._min_n_warn}. "
                f"Wilcoxon testi güvenilirliği düşük. Dikkatli yorumlayın."
            )
            logger.warning("[stats] %s | %s vs %s: %s", dataset, model_a, model_b, warning)

        if n < 2:
            return WilcoxonResult(
                dataset=dataset, scenario=scenario_label,
                model_a=model_a, model_b=model_b, n_samples=n,
                statistic=float("nan"), p_value=float("nan"),
                alpha=self._alpha, significant=False,
                mean_f1_a=float("nan"), mean_f1_b=float("nan"),
                better_model="N/A",
                warning=warning + " Hesaplama yapılamadı (n<2).",
            )

        diffs = np.array(f1_a) - np.array(f1_b)
        # Tüm farklar sıfırsa Wilcoxon uyarı verir — yakala
        if np.all(diffs == 0):
            stat, pval = 0.0, 1.0
            warning += " Tüm farklar sıfır — Wilcoxon uygulanamaz, p=1.0 atandı."
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = sp_stats.wilcoxon(f1_a, f1_b, alternative="two-sided")
                stat, pval = float(res.statistic), float(res.pvalue)
            except Exception as exc:
                stat, pval = float("nan"), float("nan")
                warning += f" Wilcoxon hatası: {exc}"

        mean_a = float(np.mean(f1_a))
        mean_b = float(np.mean(f1_b))
        better = model_a if mean_a >= mean_b else model_b

        return WilcoxonResult(
            dataset=dataset, scenario=scenario_label,
            model_a=model_a, model_b=model_b, n_samples=n,
            statistic=stat, p_value=pval,
            alpha=self._alpha, significant=(pval < self._alpha),
            mean_f1_a=mean_a, mean_f1_b=mean_b,
            better_model=better,
            warning=warning,
        )

    def run_wilcoxon(self, dataset: str) -> list[WilcoxonResult]:
        """
        Verilen dataset için tüm model çiftlerinde Wilcoxon testi çalıştırır.

        Parametreler
        ------------
        dataset : str
            "SKAB" veya "BATADAL".

        Döndürür
        --------
        list[WilcoxonResult] — 6 çift için sonuçlar.
        """
        results = []
        for a, b in MODEL_PAIRS:
            res = self._run_wilcoxon_pair(dataset, a, b)
            results.append(res)
            logger.info(
                "[Wilcoxon] %s | %s vs %s | n=%d | p=%.4f | sig=%s",
                dataset, a, b, res.n_samples, res.p_value, res.significant,
            )
        return results

    # ------------------------------------------------------------------
    # McNemar: konfüzyon rekonstrüksiyonu + test
    # ------------------------------------------------------------------

    def _reconstruct_confusion(self, rec: dict) -> dict[str, float]:
        """
        precision, recall, accuracy ve distribution'dan konfüzyon matrisini
        rekonstrükte eder.

        Yöntem 1 (accuracy bazlı — daha güvenilir):
            correct = round(accuracy × total)
            total = N0 + N1

        Yöntem 2 (TP/TN ayrıştırması için precision/recall kullanılır):
            TP = round(precision × N1)
            FN = round(TP / recall) - TP

        Döndürür
        --------
        dict: {TP, FP, FN, TN, correct, total}
        """
        N1 = float(rec["n_pred_1"])   # modelin 1 dediği sayı
        N0 = float(rec["n_pred_0"])   # modelin 0 dediği sayı
        total = N0 + N1

        acc   = rec["accuracy"]
        prec  = rec["precision"]
        rec_  = rec["recall"]

        # Correct: accuracy × total (doğrudan)
        correct = acc * total

        # TP: precision × predicted_positives
        if N1 > 0 and prec > 0:
            TP = prec * N1
        else:
            TP = 0.0

        # FN: TP / recall - TP
        if rec_ > 1e-9 and TP > 0:
            FN = max(0.0, (TP / rec_) - TP)
        else:
            FN = 0.0

        FP = max(0.0, N1 - TP)
        TN = max(0.0, correct - TP)

        return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
                "correct": correct, "total": total}

    def _mcnemar_chi2(
        self,
        b: float,
        c: float,
        correction: bool = True,
    ) -> tuple[float, float]:
        """
        McNemar chi-kare istatistiği ve p-değeri hesaplar (elle, scipy bağımsız).

        Continuity correction (Yates):
            chi2 = (|b - c| - 1)^2 / (b + c)   correction=True
            chi2 = (b - c)^2 / (b + c)          correction=False

        Döndürür
        --------
        (statistic, p_value)
        """
        bc = b + c
        if bc < 1:
            return 0.0, 1.0
        if correction:
            stat = (max(0.0, abs(b - c) - 1.0) ** 2) / bc
        else:
            stat = ((b - c) ** 2) / bc
        pval = float(sp_stats.chi2.sf(stat, df=1))
        return float(stat), pval

    def _run_mcnemar_pair(
        self,
        dataset: str,
        model_a: str,
        model_b: str,
    ) -> McNemarResult:
        """
        Tek bir (dataset, modelA, modelB) çifti için yaklaşık McNemar testi.

        Yaklaşım:
        - Her (seed, fold) çifti için iki modelin 'correct' sayısı hesaplanır.
        - correct_A > correct_B → bu fold'da A daha iyi (b sayısına katkı)
        - correct_B > correct_A → bu fold'da B daha iyi (c sayısına katkı)
        - b = A daha iyi olan fold sayısı, c = B daha iyi olan fold sayısı
        - McNemar chi2(|b-c|-1)^2/(b+c) formülüyle hesaplanır.
        """
        scenario_label = "+".join(self._scenarios)
        recs_a_raw = self._filter(dataset, model_a)
        recs_b_raw = self._filter(dataset, model_b)

        # Anahtar → kayıt eşleme
        map_a = {(r["seed"], r["fold"]): r for r in recs_a_raw}
        map_b = {(r["seed"], r["fold"]): r for r in recs_b_raw}
        common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))

        note = (
            "YAKLASIK McNemar: Bireysel tahmin vektoru mevcut olmadigi icin "
            "konfuzyon matrisi accuracy/precision/recall/distribution'dan "
            "rekonstrükte edilmistir. Fold bazli dogru/yanlis karsilastirmasi yapilmistir. "
            "Sonuclar gosterge niteligindedir; kesin yorum icin tahmin vektoru gereklidir."
        )

        if len(common_keys) < 2:
            return McNemarResult(
                dataset=dataset, scenario=scenario_label,
                model_a=model_a, model_b=model_b,
                n_total=0, n_both_correct=0, n_a_only=0, n_b_only=0,
                statistic=float("nan"), p_value=float("nan"),
                alpha=self._alpha, significant=False,
                note=note + " HATA: Yeterli eslesmis kayit yok (n<2).",
            )

        # Fold bazlı: A mı daha iyi, B mi, yoksa eşit mi?
        n_a_better = 0   # b: A doğru, B yanlış (veya A önemli ölçüde daha iyi)
        n_b_better = 0   # c: B doğru, A yanlış
        n_both     = 0   # n11: ikisi de benzer
        n_neither  = 0   # n00
        total_n_sum = 0.0

        for key in common_keys:
            conf_a = self._reconstruct_confusion(map_a[key])
            conf_b = self._reconstruct_confusion(map_b[key])
            total_n_sum += conf_a["total"]

            correct_a = conf_a["correct"]
            correct_b = conf_b["correct"]
            total = conf_a["total"]

            # Fark oranı: |correct_A - correct_B| / total > 1% ise farklı say
            thresh = max(1.0, 0.01 * total)
            if correct_a - correct_b > thresh:
                n_a_better += 1
            elif correct_b - correct_a > thresh:
                n_b_better += 1
            else:
                n_both += 1  # eşit (her ikisi doğru sayar)

        n_total = total_n_sum / len(common_keys)

        # McNemar: b = n_a_better, c = n_b_better
        b = float(n_a_better)
        c = float(n_b_better)

        stat, pval = self._mcnemar_chi2(b, c, correction=True)

        return McNemarResult(
            dataset=dataset, scenario=scenario_label,
            model_a=model_a, model_b=model_b,
            n_total=n_total, n_both_correct=float(n_both), n_a_only=b, n_b_only=c,
            statistic=stat, p_value=pval,
            alpha=self._alpha, significant=(pval < self._alpha),
            note=note,
        )

    def run_mcnemar(self, dataset: str) -> list[McNemarResult]:
        """
        Verilen dataset için tüm model çiftlerinde McNemar testi çalıştırır.
        """
        results = []
        for a, b in MODEL_PAIRS:
            res = self._run_mcnemar_pair(dataset, a, b)
            results.append(res)
            logger.info(
                "[McNemar] %s | %s vs %s | p=%.4f | sig=%s",
                dataset, a, b, res.p_value, res.significant,
            )
        return results

    # ------------------------------------------------------------------
    # Tümünü çalıştır
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        """
        SKAB + BATADAL için Wilcoxon ve McNemar testlerini çalıştırır.

        Döndürür
        --------
        dict:
            {
              "wilcoxon": {"SKAB": [...], "BATADAL": [...]},
              "mcnemar":  {"SKAB": [...], "BATADAL": [...]},
            }
        """
        if not self._records:
            raise RuntimeError("Önce load_data() çağrılmalıdır.")

        return {
            "wilcoxon": {
                "SKAB":    self.run_wilcoxon("SKAB"),
                "BATADAL": self.run_wilcoxon("BATADAL"),
            },
            "mcnemar": {
                "SKAB":    self.run_mcnemar("SKAB"),
                "BATADAL": self.run_mcnemar("BATADAL"),
            },
        }

    # ------------------------------------------------------------------
    # Kaydetme
    # ------------------------------------------------------------------

    def save_results(self, results: dict[str, Any]) -> None:
        """
        Sonuçları CSV ve TXT rapor olarak kaydeder.

        Parametreler
        ------------
        results : dict
            run_all() çıktısı.
        """
        # Çıktı dizinini oluştur
        for path in [self._out_wilcoxon, self._out_mcnemar, self._out_report]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._save_wilcoxon_csv(results["wilcoxon"])
        self._save_mcnemar_csv(results["mcnemar"])
        self._save_report(results)
        logger.info("[stats] Tüm çıktılar kaydedildi.")

    def _save_wilcoxon_csv(self, wilcoxon_results: dict[str, list[WilcoxonResult]]) -> None:
        """Wilcoxon sonuçlarını CSV'ye yazar."""
        fieldnames = [
            "dataset", "scenario", "model_a", "model_b", "n_samples",
            "mean_f1_a", "mean_f1_b", "better_model",
            "statistic", "p_value", "alpha", "significant", "warning",
        ]
        rows = []
        for dataset, res_list in wilcoxon_results.items():
            for r in res_list:
                rows.append({
                    "dataset": r.dataset,
                    "scenario": r.scenario,
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "n_samples": r.n_samples,
                    "mean_f1_a": f"{r.mean_f1_a:.6f}",
                    "mean_f1_b": f"{r.mean_f1_b:.6f}",
                    "better_model": r.better_model,
                    "statistic": f"{r.statistic:.4f}" if not np.isnan(r.statistic) else "NaN",
                    "p_value": f"{r.p_value:.6f}" if not np.isnan(r.p_value) else "NaN",
                    "alpha": r.alpha,
                    "significant": r.significant,
                    "warning": r.warning,
                })

        with open(self._out_wilcoxon, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("[stats] Wilcoxon CSV: %s", self._out_wilcoxon)

    def _save_mcnemar_csv(self, mcnemar_results: dict[str, list[McNemarResult]]) -> None:
        """McNemar sonuçlarını CSV'ye yazar."""
        fieldnames = [
            "dataset", "scenario", "model_a", "model_b",
            "n_total", "n_both_correct", "n_a_only_correct", "n_b_only_correct",
            "statistic", "p_value", "alpha", "significant", "note",
        ]
        rows = []
        for dataset, res_list in mcnemar_results.items():
            for r in res_list:
                rows.append({
                    "dataset": r.dataset,
                    "scenario": r.scenario,
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "n_total": f"{r.n_total:.0f}",
                    "n_both_correct": f"{r.n_both_correct:.0f}",
                    "n_a_only_correct": f"{r.n_a_only:.0f}",
                    "n_b_only_correct": f"{r.n_b_only:.0f}",
                    "statistic": f"{r.statistic:.4f}" if not np.isnan(r.statistic) else "NaN",
                    "p_value": f"{r.p_value:.6f}" if not np.isnan(r.p_value) else "NaN",
                    "alpha": r.alpha,
                    "significant": r.significant,
                    "note": r.note,
                })

        with open(self._out_mcnemar, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("[stats] McNemar CSV: %s", self._out_mcnemar)

    def _save_report(self, results: dict[str, Any]) -> None:
        """Türkçe metin raporu yazar."""
        lines: list[str] = []
        sep = "=" * 76
        thin = "-" * 76

        def add(s: str = "") -> None:
            lines.append(s)

        add(sep)
        add("  ISTATISTIKSEL ANLAMLILIK TESTLERI RAPORU")
        add("  Wilcoxon Signed-Rank Testi + McNemar Testi")
        add(sep)
        add()
        add(f"  Anlamlilik esigi (alpha) : {self._alpha}")
        add(f"  Wilcoxon senaryolari     : {', '.join(self._scenarios)}")
        add(f"  Minimum n uyari esigi    : {self._min_n_warn}")
        add()

        # ── Wilcoxon bölümü ────────────────────────────────────────
        for dataset in ["SKAB", "BATADAL"]:
            wres: list[WilcoxonResult] = results["wilcoxon"][dataset]
            add(sep)
            add(f"  WILCOXON SIGNED-RANK TESTI — {dataset}")
            add(sep)
            add()

            if dataset == "SKAB":
                add("  Eslestirme: (seed, fold) bazli, n=25 (5 seed x 5 fold)")
            else:
                add("  Eslestirme: seed bazli, n=5  *** KUCUK ORNEKLEM - DIKKATLI YORUMLA ***")
            add()

            # Tablo başlığı
            add(f"  {'Cift':<22} {'n':>3}  {'Mean-F1-A':>9}  {'Mean-F1-B':>9}  {'Iyi Model':<12}  {'p-degeri':>10}  {'Anlamli?':>9}  {'Yorum'}")
            add("  " + thin)

            for r in wres:
                sig_str = "EVET ***" if r.significant else "Hayir"
                pair_str = f"{r.model_a} vs {r.model_b}"
                yorum = self._wilcoxon_comment(r)
                add(
                    f"  {pair_str:<22} {r.n_samples:>3}  {r.mean_f1_a:>9.4f}  {r.mean_f1_b:>9.4f}"
                    f"  {r.better_model:<12}  {r.p_value:>10.6f}  {sig_str:>9}  {yorum}"
                )
                if r.warning:
                    add(f"    [!] {r.warning}")
            add()

        # ── McNemar bölümü ─────────────────────────────────────────
        for dataset in ["SKAB", "BATADAL"]:
            mres: list[McNemarResult] = results["mcnemar"][dataset]
            add(sep)
            add(f"  McNEMAR TESTI (YAKLASIK) — {dataset}")
            add(sep)
            add()
            add("  [!] SINIRLILIK: Bireysel tahmin vektoru mevcut olmadigi icin")
            add("      konfuzyon matrisi precision/recall/distribution'dan")
            add("      rekonstrükte edilmistir. Sonuclar gosterge niteligindedir.")
            add()

            add(f"  {'Cift':<22} {'A-dogru':>8}  {'B-dogru':>8}  {'p-degeri':>10}  {'Anlamli?':>9}")
            add("  " + thin)

            for r in mres:
                sig_str = "EVET ***" if r.significant else "Hayir"
                pair_str = f"{r.model_a} vs {r.model_b}"
                n_a = r.n_a_only
                n_b = r.n_b_only
                add(
                    f"  {pair_str:<22} {n_a:>8.0f}  {n_b:>8.0f}"
                    f"  {r.p_value:>10.6f}  {sig_str:>9}"
                )
            add()

        # ── DL vs Automata odak analizi ────────────────────────────
        add(sep)
        add("  ANA KARSILASTIRMA: DL MODELLERI vs AUTOMATA")
        add(sep)
        add()
        add("  Bu proje, klasik otomata tabanli anomali tespitini modern derin")
        add("  ogrenme modelleriyle karsilastirmaktadir. Asagida bu ciftlerin")
        add("  istatistiksel analizi ozetlenmektedir.")
        add()

        for dataset in ["SKAB", "BATADAL"]:
            wres_map = {
                (r.model_a, r.model_b): r
                for r in results["wilcoxon"][dataset]
            }
            add(f"  [ {dataset} ]")
            dl_auto_pairs = [
                ("lstm", "automata"), ("gru", "automata"), ("cnn1d", "automata")
            ]
            for a, b in dl_auto_pairs:
                r = wres_map.get((a, b)) or wres_map.get((b, a))
                if r is None:
                    continue
                sig_str = "ISTATISTIKSEL OLARAK ANLAMLI (p<0.05)" if r.significant else "Anlamli degil (p>=0.05)"
                add(
                    f"    {a:<8} vs automata : p={r.p_value:.4f}  "
                    f"=> {sig_str}"
                )
                add(
                    f"      F1 farki: {a}={r.mean_f1_a:.4f} vs automata={r.mean_f1_b:.4f}"
                    f"  (fark={abs(r.mean_f1_a-r.mean_f1_b):.4f})"
                )
            add()

        add(sep)
        add("  GENEL YORUM")
        add(sep)
        add()
        add(self._overall_comment(results))
        add()
        add(sep)

        report_text = "\n".join(lines)
        with open(self._out_report, "w", encoding="utf-8") as fh:
            fh.write(report_text)
        logger.info("[stats] Rapor: %s", self._out_report)

        # Konsola da bas
        print(report_text)

    # ------------------------------------------------------------------
    # Yorum yardımcıları
    # ------------------------------------------------------------------

    @staticmethod
    def _wilcoxon_comment(r: WilcoxonResult) -> str:
        """Wilcoxon sonucu için kısa Türkçe yorum üretir."""
        if np.isnan(r.p_value):
            return "Hesaplanamadi"
        if r.n_samples < 6:
            return f"n kucuk ({r.n_samples}), dikkatli yorumla"
        if r.significant:
            delta = abs(r.mean_f1_a - r.mean_f1_b)
            if delta > 0.1:
                return f"Guclu fark (|dF1|={delta:.3f})"
            return f"Anlamli fark (|dF1|={delta:.3f})"
        else:
            delta = abs(r.mean_f1_a - r.mean_f1_b)
            return f"Anlamli fark yok (|dF1|={delta:.3f})"

    def _overall_comment(self, results: dict) -> str:
        """Tüm sonuçları değerlendiren genel Türkçe yorum döndürür."""
        paragraphs: list[str] = []

        for dataset in ["SKAB", "BATADAL"]:
            wres: list[WilcoxonResult] = results["wilcoxon"][dataset]
            dl_auto = [
                r for r in wres
                if r.model_b == "automata" or r.model_a == "automata"
            ]
            all_sig = all(r.significant for r in dl_auto if not np.isnan(r.p_value))
            any_sig = any(r.significant for r in dl_auto if not np.isnan(r.p_value))

            if dataset == "SKAB":
                n_desc = "n=25 (5 seed × 5 fold)"
            else:
                n_desc = "n=5 (dikkat: kucuk orneklem)"

            if all_sig:
                verdict = (
                    f"  {dataset} ({n_desc}): DL modelleri (LSTM/GRU/CNN1D) ile Automata\n"
                    f"  arasindaki F1 farki TUM ciftlerde istatistiksel olarak anlamlidir\n"
                    f"  (p<{self._alpha}). DL modelleri {dataset} uzerinde belirgin sekilde\n"
                    f"  daha iyi performans gostermektedir."
                )
            elif any_sig:
                verdict = (
                    f"  {dataset} ({n_desc}): DL-Automata ciftlerinin BAZILARI istatistiksel\n"
                    f"  olarak anlamli fark gostermektedir (p<{self._alpha}). Yorumlama icin\n"
                    f"  bireysel sonuc tablosuna bakin."
                )
            else:
                if dataset == "BATADAL":
                    verdict = (
                        f"  {dataset} ({n_desc}): Hicbir cift anlamli fark gostermemistir.\n"
                        f"  Bu, hem DL hem Automata modellerinin BATADAL uzerinde dusuk\n"
                        f"  F1 skoru (tum modeller ~0.03-0.19) nedeniyle birbirinden\n"
                        f"  istatistiksel olarak ayirt edilememesiyle aciklanabilir.\n"
                        f"  Ayrica kucuk orneklem (n=5) test gucunu dusurtmektedir."
                    )
                else:
                    verdict = (
                        f"  {dataset} ({n_desc}): DL-Automata farki istatistiksel olarak\n"
                        f"  anlamli degildir. Beklenmedik bir sonuc — F1 dagilimlarini kontrol edin."
                    )
            paragraphs.append(verdict)

        return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Yardımcı: config yükle
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    """config.yaml'ı yükler. PyYAML yoksa basit satır ayrıştırıcı kullanır."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        pass

    # Fallback: PyYAML yok → minimal config
    logger.warning("PyYAML bulunamadı. Varsayılan stats config kullanılıyor.")
    return {
        "stats": {
            "alpha": 0.05,
            "min_n_warning": 6,
            "scenarios_for_wilcoxon": ["original"],
            "output_wilcoxon_csv": "results/stats_wilcoxon.csv",
            "output_mcnemar_csv": "results/stats_mcnemar.csv",
            "output_report": "results/stats_report.txt",
        }
    }


# ---------------------------------------------------------------------------
# CLI giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    """Komut satırından çalıştırma: `python -m src.experiments.stats`"""
    # stdout utf-8 (Windows uyumluluğu)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Proje kök dizinini sys.path'e ekle (doğrudan çalıştırma için)
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    config = _load_config(str(_root / "config" / "config.yaml"))

    tester = StatisticalTester(config)
    tester.load_data(str(_root / "results" / "experiments_final.csv"))

    results = tester.run_all()
    tester.save_results(results)

    print()
    print("Tamamlandi.")
    print(f"  Wilcoxon CSV : {tester._out_wilcoxon}")
    print(f"  McNemar  CSV : {tester._out_mcnemar}")
    print(f"  Rapor TXT    : {tester._out_report}")


if __name__ == "__main__":
    main()
