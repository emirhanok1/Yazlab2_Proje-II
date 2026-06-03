"""
test_unseen.py — UnseenHandler + edit_distance birim testleri (pytest).

Calistirma:
    python -m pytest tests/test_unseen.py -v

Test gruplari:
  TestEditDistance     — edit_distance() fonksiyonu icin dogru hesap
  TestUnseenHandler    — lookup() seen/unseen siniflandirmasi
  TestNearestNeighbor  — en yakin pattern bulma (ornk. 'adc' -> 'abc' dist=1)
  TestDetectionRate    — detection_rate() ve mapping_report()
  TestEdgeCases        — bos/tek eleman/ayni string edge case'ler
  TestIntegration      — ProbabilisticAutomata ile entegrasyon
"""

import sys
import os
import pytest

# Proje kokunu path'e ekle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.automata.unseen import edit_distance, UnseenHandler, LookupResult


# ---------------------------------------------------------------------------
# Sabit test verisi
# ---------------------------------------------------------------------------

# Kucuk bir train state sozlugu (pattern -> state_id)
TRAIN_VOCAB = {
    "abc": 0,
    "bca": 1,
    "cab": 2,
    "aab": 3,
    "bbc": 4,
    "cca": 5,
}

# 'adc' -> 'abc' distance=1  (d->b)
# 'xyz' -> en kisa mevcut; hepsine uzak ama en yakini 'bbc' veya 'abc' gibi
# 'abcx' -> farkli uzunluk


# ===========================================================================
# 1. Edit Distance Hesabi
# ===========================================================================

class TestEditDistance:
    """edit_distance() fonksiyonu dogru Levenshtein mesafesi vermeli."""

    def test_identical_strings(self):
        """Ayni string -> mesafe 0."""
        assert edit_distance("abc", "abc") == 0

    def test_single_substitution(self):
        """Tek karakter degisimi -> mesafe 1."""
        assert edit_distance("abc", "axc") == 1

    def test_single_insertion(self):
        """Tek karakter ekleme -> mesafe 1."""
        assert edit_distance("abc", "abbc") == 1

    def test_single_deletion(self):
        """Tek karakter silme -> mesafe 1."""
        assert edit_distance("abbc", "abc") == 1

    def test_adc_to_abc(self):
        """'adc' -> 'abc': tek substitution (d->b) -> distance=1."""
        assert edit_distance("adc", "abc") == 1

    def test_empty_strings(self):
        """Her iki string bos -> distance=0."""
        assert edit_distance("", "") == 0

    def test_one_empty(self):
        """Biri bos -> diger uzunluğu kadar distance."""
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_completely_different(self):
        """Hic ortak karakter yok -> tam degisim."""
        # 'aaa' -> 'bbb': 3 substitution
        assert edit_distance("aaa", "bbb") == 3

    def test_symmetry(self):
        """Levenshtein simetrik olmali: d(a,b) == d(b,a)."""
        pairs = [
            ("abc", "adc"),
            ("hello", "helo"),
            ("xyz", "abc"),
            ("", "ab"),
        ]
        for a, b in pairs:
            assert edit_distance(a, b) == edit_distance(b, a), (
                f"Simetri ihlali: d({a!r},{b!r})={edit_distance(a,b)} "
                f"!= d({b!r},{a!r})={edit_distance(b,a)}"
            )

    def test_triangle_inequality(self):
        """Ucgen esitsizligi: d(a,c) <= d(a,b) + d(b,c)."""
        a, b, c = "abc", "axc", "xyz"
        assert edit_distance(a, c) <= edit_distance(a, b) + edit_distance(b, c)

    def test_known_pairs(self):
        """Dogrulugu bilinen spesifik ciftler."""
        cases = [
            ("kitten", "sitting", 3),
            ("saturday", "sunday", 3),
            ("a",       "b",      1),
            ("ab",      "ba",     2),   # transpose = 2 swap
        ]
        for a, b, expected in cases:
            result = edit_distance(a, b)
            assert result == expected, (
                f"edit_distance({a!r},{b!r}) = {result}, beklenen {expected}"
            )


# ===========================================================================
# 2. Seen / Unseen Siniflandirmasi
# ===========================================================================

class TestUnseenHandler:
    """lookup() metodu seen/unseen'i dogru siniflandirmali."""

    @pytest.fixture
    def handler(self):
        return UnseenHandler(TRAIN_VOCAB)

    def test_seen_pattern_not_marked_unseen(self, handler):
        """Bilinen pattern unseen DEĞİL."""
        for pattern in TRAIN_VOCAB:
            result = handler.lookup(pattern)
            assert not result.is_unseen, (
                f"Bilinen pattern {pattern!r} yanlis sekilde unseen isaretlendi."
            )

    def test_seen_pattern_distance_zero(self, handler):
        """Bilinen pattern icin edit distance = 0."""
        for pattern in TRAIN_VOCAB:
            result = handler.lookup(pattern)
            assert result.distance == 0, (
                f"Bilinen {pattern!r}: distance={result.distance}, beklenen 0."
            )

    def test_seen_resolved_equals_query(self, handler):
        """Seen pattern: resolved == query."""
        for pattern in TRAIN_VOCAB:
            result = handler.lookup(pattern)
            assert result.resolved == result.query

    def test_seen_state_id_correct(self, handler):
        """Seen pattern'in state_id dogru eslenmeli."""
        for pattern, expected_id in TRAIN_VOCAB.items():
            result = handler.lookup(pattern)
            assert result.state_id == expected_id, (
                f"{pattern!r}: state_id={result.state_id}, beklenen {expected_id}"
            )

    def test_unseen_pattern_marked(self, handler):
        """Sozlukte olmayan pattern unseen olarak isaretlenmeli."""
        unseen_patterns = ["xyz", "zzz", "adc", "qwe", "aaaabc"]
        for pattern in unseen_patterns:
            result = handler.lookup(pattern)
            assert result.is_unseen, (
                f"Unseen {pattern!r} yanlis sekilde seen olarak isaretlendi."
            )

    def test_unseen_distance_positive(self, handler):
        """Unseen pattern icin distance >= 1."""
        unseen_patterns = ["xyz", "adc", "zzz"]
        for pattern in unseen_patterns:
            result = handler.lookup(pattern)
            assert result.distance >= 1, (
                f"Unseen {pattern!r}: distance={result.distance}, en az 1 olmali."
            )

    def test_unseen_resolved_in_vocab(self, handler):
        """Unseen pattern'in resolved degeri train sozlugunde olmali."""
        result = handler.lookup("adc")
        assert result.resolved in TRAIN_VOCAB, (
            f"resolved={result.resolved!r} train sozlugunde degil."
        )

    def test_lookup_result_type(self, handler):
        """lookup() her zaman LookupResult dondurmeli."""
        result = handler.lookup("abc")
        assert isinstance(result, LookupResult)

    def test_is_seen_helper(self, handler):
        """is_seen() yardimci metodu dogru calismal."""
        for p in TRAIN_VOCAB:
            assert handler.is_seen(p), f"{p!r} seen olmali."
        for p in ["xyz", "adc", "qqq"]:
            assert not handler.is_seen(p), f"{p!r} unseen olmali."


# ===========================================================================
# 3. Nearest-Neighbor Mapping
# ===========================================================================

class TestNearestNeighbor:
    """Levenshtein tabanli en yakin pattern bulma."""

    @pytest.fixture
    def handler(self):
        return UnseenHandler(TRAIN_VOCAB)

    def test_adc_maps_to_abc(self, handler):
        """'adc' -> 'abc' (tek substitution, distance=1)."""
        handler.reset_log()
        result = handler.lookup("adc")
        assert result.is_unseen
        assert result.resolved == "abc", (
            f"'adc' -> '{result.resolved}' beklenen 'abc'"
        )
        assert result.distance == 1

    def test_bba_maps_correctly(self, handler):
        """'bba' en yakin: 'bbc' veya 'bca' gibi. Distance=1 olmali."""
        # 'bba' -> 'bbc': b-b-a vs b-b-c => 1 substitution
        # 'bba' -> 'bca': b-b-a vs b-c-a => 1 substitution
        # Alfabetik kucuk olan secilir
        handler.reset_log()
        result = handler.lookup("bba")
        assert result.is_unseen
        assert result.distance == 1
        # Esit uzaklikta: 'bbc' < 'bca' alfabetik -> 'bbc' secilmeli
        assert result.resolved == "bbc", (
            f"'bba' -> '{result.resolved}' beklenen 'bbc' (alfabetik kucuk)"
        )

    def test_nearest_is_minimum_distance(self, handler):
        """Secilen nearest pattern, tum adaylarin minimum mesafesinde olmali."""
        test_unseen = ["adc", "bba", "xyz", "ccc"]
        for query in test_unseen:
            if handler.is_seen(query):
                continue
            result = handler.lookup(query)
            # Tum bilinenlerin mesafesini hesapla
            all_dists = {p: edit_distance(query, p) for p in TRAIN_VOCAB}
            min_dist = min(all_dists.values())
            assert result.distance == min_dist, (
                f"'{query}': result.distance={result.distance}, "
                f"gercek min={min_dist}, adaylar={all_dists}"
            )

    def test_deterministic_tie_breaking(self, handler):
        """Esit mesafeli adaylar arasindan her zaman ayni (alfabetik kucuk) secilmeli."""
        # Ayni sorguyu iki kez calistir -> ayni sonuc
        handler.reset_log()
        r1 = handler.lookup("adc")
        handler.reset_log()
        r2 = handler.lookup("adc")
        assert r1.resolved == r2.resolved, "Ayni sorgu -> ayni resolved (deterministik)"

    def test_single_char_vocab(self):
        """Tek patternlik sozlukte her sorgu ona map'lenmeli."""
        vocab = {"abc": 0}
        h = UnseenHandler(vocab)
        result = h.lookup("xyz")
        assert result.resolved == "abc"
        assert result.is_unseen

    def test_different_length_unseen(self, handler):
        """Farkli uzunlukta unseen pattern de dogru map'lenmeli."""
        handler.reset_log()
        result = handler.lookup("ab")   # 2 karakter, vocab'da 3 karakter var
        assert result.is_unseen
        assert result.distance >= 1
        assert result.resolved in TRAIN_VOCAB


# ===========================================================================
# 4. Detection Rate ve Mapping Report
# ===========================================================================

class TestDetectionRate:
    """detection_rate() ve mapping_report() istatistikleri."""

    def test_all_seen_rate_zero(self):
        """Tum pattern'lar bilinen ise detection_rate = 0."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(list(TRAIN_VOCAB.keys()))
        assert h.detection_rate() == 0.0

    def test_all_unseen_rate_one(self):
        """Hic bilinmeyen pattern ise detection_rate = 1."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["xyz", "qqq", "rrr"])
        assert h.detection_rate() == 1.0

    def test_mixed_detection_rate(self):
        """Karisik: 2 seen + 2 unseen -> rate = 0.5."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["abc", "bca", "xyz", "qqq"])
        assert h.detection_rate() == pytest.approx(0.5)

    def test_empty_log_rate_zero(self):
        """Hic sorgu yapilmamissa detection_rate = 0."""
        h = UnseenHandler(TRAIN_VOCAB)
        assert h.detection_rate() == 0.0

    def test_mapping_report_keys(self):
        """mapping_report() zorunlu anahtarlari icermeli."""
        required_keys = {
            "total_queries", "n_seen", "n_unseen",
            "detection_rate", "avg_distance", "max_distance", "mappings",
        }
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["abc", "adc"])
        report = h.mapping_report()
        assert required_keys <= set(report.keys()), (
            f"Eksik anahtarlar: {required_keys - set(report.keys())}"
        )

    def test_mapping_report_counts(self):
        """mapping_report() n_seen + n_unseen == total_queries."""
        h = UnseenHandler(TRAIN_VOCAB)
        patterns = ["abc", "bca", "xyz", "adc", "cab"]
        h.score_sequence(patterns)
        report = h.mapping_report()
        assert report["total_queries"] == len(patterns)
        assert report["n_seen"] + report["n_unseen"] == report["total_queries"]

    def test_mapping_report_avg_distance_nonneg(self):
        """avg_distance >= 0."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["abc", "xyz", "adc"])
        report = h.mapping_report()
        assert report["avg_distance"] >= 0.0

    def test_mapping_report_only_unseen_in_mappings(self):
        """mappings listesi sadece unseen sonuclari icermeli."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["abc", "xyz"])
        report = h.mapping_report()
        assert report["n_unseen"] == len(report["mappings"])

    def test_reset_log_clears_state(self):
        """reset_log() sonrasi detection_rate sifirlanmali."""
        h = UnseenHandler(TRAIN_VOCAB)
        h.score_sequence(["xyz", "qqq"])
        assert h.detection_rate() == 1.0
        h.reset_log()
        assert h.detection_rate() == 0.0

    def test_score_sequence_returns_results(self):
        """score_sequence() her pattern icin LookupResult dondurmeli."""
        h = UnseenHandler(TRAIN_VOCAB)
        patterns = ["abc", "xyz", "bca"]
        results = h.score_sequence(patterns)
        assert len(results) == len(patterns)
        assert all(isinstance(r, LookupResult) for r in results)


# ===========================================================================
# 5. Edge Case'ler
# ===========================================================================

class TestEdgeCases:
    """Bos, tek eleman, cok uzun string gibi sinir durumlar."""

    def test_empty_state_index_raises(self):
        """Bos state_index ile UnseenHandler olusturma -> ValueError."""
        with pytest.raises(ValueError):
            UnseenHandler({})

    def test_single_pattern_vocab(self):
        """Tek patternlik sozluk: her sorgu ona map'lenmeli."""
        h = UnseenHandler({"abc": 0})
        result = h.lookup("xyz")
        assert result.resolved == "abc"
        assert result.state_id == 0

    def test_empty_string_query(self):
        """Bos string sorgusu: en kisa pattern'e map'lenmeli."""
        h = UnseenHandler(TRAIN_VOCAB)
        result = h.lookup("")
        # En yakin: en kisa pattern (distance = uzunluk)
        assert result.is_unseen
        assert result.resolved in TRAIN_VOCAB

    def test_exact_match_not_unseen(self):
        """Tam eslesme: is_unseen=False ve distance=0."""
        h = UnseenHandler(TRAIN_VOCAB)
        result = h.lookup("cca")
        assert not result.is_unseen
        assert result.distance == 0
        assert result.state_id == TRAIN_VOCAB["cca"]

    def test_lookup_does_not_modify_vocab(self):
        """lookup() train sozlugunu degistirmemeli (leakage yok)."""
        h = UnseenHandler(TRAIN_VOCAB)
        before = dict(h._state_index)
        h.lookup("newpattern")
        h.lookup("abc")
        after = dict(h._state_index)
        assert before == after, "lookup() sozlugu degistirdi!"

    def test_known_patterns_sorted(self):
        """known_patterns alfabetik sirali olmali."""
        h = UnseenHandler(TRAIN_VOCAB)
        kp = h.known_patterns
        assert kp == sorted(kp), "known_patterns alfabetik sirali degil."

    def test_n_known_patterns(self):
        """n_known_patterns sozluk boyutuna esit olmali."""
        h = UnseenHandler(TRAIN_VOCAB)
        assert h.n_known_patterns == len(TRAIN_VOCAB)


# ===========================================================================
# 6. Entegrasyon: ProbabilisticAutomata ile birlikte
# ===========================================================================

class TestIntegration:
    """UnseenHandler'in gercek ProbabilisticAutomata state_index_ ile calismasi."""

    @pytest.fixture
    def fitted_model_and_handler(self):
        """Kucuk sentetik veriyle fit edilmis model + handler."""
        from src.models.automata.automata import ProbabilisticAutomata, _symbols_to_patterns
        from src.utils.config import load_config

        config = load_config(os.path.join(PROJECT_ROOT, "config", "config.yaml"))

        # Deterministik kucuk SAX dizisi (a,b,c alfabesiyle)
        import numpy as np
        rng = np.random.default_rng(42)
        choices = ["a", "b", "c"]
        train_symbols = [rng.choice(choices) for _ in range(50)]

        model = ProbabilisticAutomata(config)
        model.fit(train_symbols)

        handler = UnseenHandler(model.state_index_)
        return model, handler, train_symbols

    def test_all_train_patterns_are_seen(self, fitted_model_and_handler):
        """Train'den gelen tum pattern'lar seen olmali."""
        from src.models.automata.automata import _symbols_to_patterns
        model, handler, train_symbols = fitted_model_and_handler
        config_window = model.window_size
        train_patterns = _symbols_to_patterns(train_symbols, config_window)
        for p in train_patterns:
            assert handler.is_seen(p), f"Train pattern {p!r} seen degil!"

    def test_fabricated_pattern_is_unseen(self, fitted_model_and_handler):
        """Uydurulmus pattern unseen olmali."""
        model, handler, _ = fitted_model_and_handler
        # window_size kadar 'z' karakterinden olusan pattern kesinlikle unseen
        fake = "z" * model.window_size
        result = handler.lookup(fake)
        assert result.is_unseen, f"{fake!r} unseen isaretlenmedi!"
        assert result.resolved in model.state_index_

    def test_state_id_valid(self, fitted_model_and_handler):
        """Resolved pattern'in state_id gecerli aralikta olmali."""
        model, handler, _ = fitted_model_and_handler
        fake = "z" * model.window_size
        result = handler.lookup(fake)
        assert 0 <= result.state_id < model.n_states, (
            f"state_id={result.state_id} gecersiz; n_states={model.n_states}"
        )

    def test_detection_rate_range(self, fitted_model_and_handler):
        """detection_rate() [0,1] araliginda olmali."""
        model, handler, train_symbols = fitted_model_and_handler
        # Yari seen, yari unseen karisiyor
        mixed = train_symbols[:5] + ["z" * model.window_size] * 3
        from src.models.automata.automata import _symbols_to_patterns
        patterns = _symbols_to_patterns(mixed, model.window_size)
        handler.score_sequence(patterns)
        dr = handler.detection_rate()
        assert 0.0 <= dr <= 1.0, f"detection_rate={dr} aralik disi!"
