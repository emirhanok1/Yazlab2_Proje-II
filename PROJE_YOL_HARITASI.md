# YazLab 2 — Proje Yol Haritası ve Görev Dağılımı
## From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

**Ekip:** Emirhan & Efekan
**Son Teslim:** 7 Haziran 2026, 23:59 (bu belge: 1 Haziran 2026 — **6 gün**)
**Hedef:** Rubrikteki 100 puanın tamamı. Ek puan (opsiyonel) analizler çekirdek bittikten sonra.

> Bu belge hem yol haritası hem de **hazır Antigravity prompt deposudur.** Her fazın altında: ne yapılacak, kim sorumlu, kopyala-yapıştır prompt, ve faz bitiş (doğrulama) kriteri vardır. Sohbet ederek ilerliyoruz; promptlar referans olarak burada durur.

---

## 0. Özet ve Mimari Kararlar (Sabitlendi)

Bu projede iki modelleme paradigması karşılaştırılıyor: **black-box derin öğrenme** (LSTM, GRU, 1D-CNN) ve **yorumlanabilir olasılıksal otomata** (PAA → SAX → sliding window → durum geçiş olasılıkları). İki veri seti (SKAB, BATADAL) üzerinde, anomali tespiti problemi olarak ele alınıyor.

**Araştırma sonrası kesinleşen 5 teknik karar:**

1. **SAX/PAA kütüphanesi: `pyts`.** `strategy='quantile'` ile breakpoint'ler yalnızca `fit(X_train)`'den öğrenilir, `transform`'da donmuş eşikler kullanılır → data leakage mimari seviyede engellenir. sklearn uyumlu.
2. **window → label kuralı: "Last-Step"** (pencerenin son satırının etiketi). Hem DL (LSTM/GRU Many-to-One) hem otomata için **aynı kural** → deneyler karşılaştırılabilir. Nedensel, alarm fırtınası yok, gerçekçi point-wise değerlendirme.
3. **Otomata path probability → satır kararı:** Sliding window üzerinde path probability hesaplanır; düşük olasılık = anomali. Çekirdekte last-step hizalama. (Mean-Aggregation ek puan/opsiyonel.)
4. **Path probability hesabı: log-space + uzunluk normalizasyonu** (per-transition average log-likelihood = geometrik ortalama). Underflow'u engeller, güven skoru bundan türer. Eşik: train dağılımının percentile'ı.
5. **Geçiş olasılığı smoothing: Add-k (Lidstone)**, k≈0.01–0.1 (validation'da ayarlanır). log(0) = −∞ çöküşünü engeller. k merkezi config'de parametre.

**Veri özeti (doğrulandı):**
- SKAB: `data/skab/valve1/` (16 csv) + `data/skab/valve2/` (4 csv) = 20 dosya. 11 sütun: datetime + 8 sensör + `anomaly`(0/1) + `changepoint`. Ayraç `;`, ondalık `,`. → 20 grup, GroupKFold için ideal.
- BATADAL: `data/batadal/BATADAL_dataset04.csv` (Training Dataset 2). 4177 satır, 45 sütun. Etiket `ATT_FLAG` ∈ {-999, 1}. Sütun isimlerinde **baştaki boşluk** (`.str.strip()` şart). Zaman: `DATETIME` (saatlik). **Map: -999 → 0 (normal), 1 → 1 (anomali). ~%5.2 anomali → class imbalance.**

---

## 1. Teknoloji Yığını

**Dil:** Python 3.11+

**Çekirdek kütüphaneler:**
- `numpy`, `pandas` — veri işleme
- `scikit-learn` — PCA, StandardScaler, GroupKFold/StratifiedGroupKFold, metrikler
- `pyts` — PAA + SAX (leakage-free, quantile strategy)
- `torch` (PyTorch) — LSTM, GRU, 1D-CNN
- `scipy` — Wilcoxon, McNemar (istatistiksel testler); `scipy.spatial.distance` veya `python-Levenshtein` — edit distance
- `matplotlib` + `seaborn` — görselleştirme (confusion matrix, ROC/PR, heatmap, state diagram)
- `networkx` veya `graphviz` — automata state diagram
- `pyyaml` — merkezi config (YAML)
- `pytest` — birim testler (unseen/Levenshtein mekanizması zorunlu test)

**Mimari prensipler (rubrik 20 puan):**
- Tüm parametreler tek bir `config.yaml`'da. Hard-coded değer YOK.
- Pipeline yapısı: `data → preprocess → model → evaluate → report` modülleri.
- Parametre değişince sistem otomatik yeniden üretir (config-driven).
- Deney takibi: her run'ın parametreleri + metrikleri otomatik loglanır (JSON/CSV).

**Önerilen repo iskeleti:**
```
Yazlab2_Proje-II/
├── config/
│   └── config.yaml          # TÜM parametreler
├── data/
│   ├── skab/{valve1,valve2}/
│   └── batadal/
├── src/
│   ├── data/                # yükleme, ön işleme, split
│   │   ├── loaders.py
│   │   ├── preprocess.py    # normalizasyon, PCA, windowing
│   │   └── splits.py        # GroupKFold (SKAB), zaman sıralı (BATADAL)
│   ├── models/
│   │   ├── deep/            # lstm.py, gru.py, cnn1d.py
│   │   └── automata/        # paa_sax.py, automata.py, unseen.py
│   ├── explain/             # açıklanabilirlik modülü
│   ├── experiments/         # senaryolar, seed döngüsü, istatistik testler
│   ├── eval/                # metrikler, görselleştirme
│   └── utils/               # logging, config loader, seed
├── tests/                   # pytest (unseen/Levenshtein zorunlu)
├── results/                 # loglar, figürler, tablolar
├── README.md                # rapor (Markdown)
├── requirements.txt
└── .gitignore
```

---

## 2. Görev Dağılımı (Genel Çerçeve)

İkiniz de hem DL hem otomata tarafına commit atmalısınız (hoca commit dağılımına bakıyor). Ana sorumluluklar:

| Alan | Sorumlu |
|---|---|
| Veri pipeline (yükleme, ön işleme, PCA, split, windowing) | **Emirhan** |
| Derin öğrenme modelleri (LSTM, GRU, 1D-CNN) | **Emirhan** |
| Otomata modeli (PAA/SAX, geçiş olasılıkları, Add-k, path prob) | **Efekan** |
| Unseen yönetimi (Levenshtein) + birim testler | **Efekan** |
| Açıklanabilirlik modülü | **Efekan** |
| Merkezi config + pipeline iskeleti + logging | **Ortak** (önce Emirhan kurar) |
| Deney protokolü (5 seed, 3 senaryo) + istatistik testler | **Ortak** |
| Görselleştirme + rapor (README) | **Ortak** |

> Not: Efekan'ın otomata + açıklanabilirlik yükü ağır; çekirdek bitiminde yük yeniden dengelenebilir. Emirhan veri+DL'i erken bitirirse otomata görselleştirme/testlerine geçer.

---

## 3. Faz Faz Yol Haritası + Antigravity Promptları

> Çalışma akışı: Sen promptu Antigravity'e atarsın → planını verir → bana gösterirsin → onaylarsam yapar. Her faz sonunda doğrulama kriterini kontrol ederiz.

---

### FAZ 0 — Repo İskeleti, Config, .gitignore (Gün 1, bugün)
**Sorumlu:** Ortak (Emirhan kurar, Efekan review). **Tahmini süre:** 2-3 saat.

**Ne yapılacak:** Klasör yapısı, `config.yaml` taslağı, `requirements.txt`, `.gitignore`, logging ut' ı, seed yönetimi. Kod yok, sadece iskelet + config.

#### Antigravity Promptu — .gitignore
```
GÖREV: Projeye kapsamlı bir .gitignore oluştur.

KURALLAR:
- Python standart: __pycache__/, *.pyc, .venv/, venv/, *.egg-info/, .pytest_cache/
- IDE: .vscode/, .idea/, *.swp
- ML/deney çıktıları: results/ altındaki büyük figür ve model ağırlıkları (*.pt, *.pth, *.ckpt), 
  ama results/ klasör yapısının kendisi (.gitkeep ile) korunsun.
- ÖNEMLİ: data/ klasöründeki ham veri setlerini (SKAB csv'leri, BATADAL csv) commit ETME 
  (büyük dosyalar). data/ altına .gitkeep koy ama *.csv'leri ignore et.
  ANCAK: veri indirme scriptini commit et ki başkası indirebilsin.
- Notebook checkpoint: .ipynb_checkpoints/
- OS: .DS_Store, Thumbs.db

ÇIKTI: .gitignore dosyasını oluştur ve içeriğini göster. data/ ignore mantığını bana açıkla.
```

#### Antigravity Promptu — Repo iskeleti + config
```
GÖREV: Projenin modüler iskeletini ve merkezi konfigürasyonunu oluştur. ŞİMDİLİK KOD YAZMA, 
sadece klasör yapısı + boş modül dosyaları (docstring'li) + config.yaml + requirements.txt.

KLASÖR YAPISI:
[buraya belgenin "Önerilen repo iskeleti" bölümünü yapıştır]

config/config.yaml İÇERİĞİ (tüm parametreler merkezi, hard-coded yasak):
- seeds: [42, 123, 2026, 7, 999]
- training: {epochs: 50, batch_size: 32, early_stopping_patience: 5}
- automata: {window_size: 4, alphabet_size: 3, smoothing_k: 0.1}
- automata_param_sweep: {window_size: [3,4,5,6], alphabet_size: [3,4,5,6]}
- preprocessing: {pca_components: 1, normalization: standard}
- split: {skab: groupkfold, batadal: {train: 0.6, val: 0.2, test: 0.2}}
- scenarios: [original, gaussian_noise, unseen]
- noise: {type: gaussian, std: <parametrik>}
- paths: {skab_valve1, skab_valve2, batadal, results}

requirements.txt: numpy, pandas, scikit-learn, pyts, torch, scipy, python-Levenshtein, 
matplotlib, seaborn, networkx, pyyaml, pytest

KISITLAR:
- Her modül dosyası bir docstring ile ne yapacağını belirtsin (implementasyon boş bırakılabilir / pass).
- config loader (src/utils/config.py) ve seed setter (src/utils/seed.py) için temel iskelet yaz.
- Hiçbir parametre koda gömülmesin; her şey config.yaml'dan okunacak şekilde tasarla.

ÇIKTI: Tüm klasör/dosya ağacını (tree) ve config.yaml içeriğini göster.
```

**Doğrulama kriteri:** Klasör ağacı oluştu, config.yaml tüm parametreleri içeriyor, requirements kuruldu (`pip install -r requirements.txt` hatasız), ilk commit atıldı (Emirhan + Efekan ikisi de commit görmeli).

---

### FAZ 1 — Veri Yükleme ve Ön İşleme (Gün 1-2)
**Sorumlu:** Emirhan. **Tahmini süre:** yarım gün.

**Ne yapılacak:** SKAB (20 csv concat + source_group/source_file kolonları) ve BATADAL yükleme. Sütun temizliği (BATADAL boşluk strip). Etiket map (BATADAL -999→0). Normalizasyon (yalnız train'de fit). PCA (yalnız train'de fit, PC1). Leakage kurallarına tam uyum.

**Kritik kurallar (rapordan):**
- SKAB: model girdisi = 8 sensör. datetime, changepoint, source_group, source_file girdiye DAHİL DEĞİL.
- SKAB ek kolonlar: `source_group` (valve1/valve2), `source_file` (hangi csv) — sadece split/takip için.
- BATADAL: model girdisi = 43 sensör/sistem değişkeni. DATETIME girdiye dahil değil (sadece sıra için).
- Normalizasyon + PCA + SAX sözlüğü + geçiş olasılıkları → YALNIZCA train'de fit.

#### Antigravity Promptu — Veri yükleme + ön işleme
```
GÖREV: src/data/ altında veri yükleme ve ön işleme modüllerini yaz. config.yaml'dan parametre oku, 
hard-code yapma.

1. loaders.py:
   - load_skab(): valve1 (16 csv) + valve2 (4 csv) hepsini oku (sep=';', decimal=','). 
     Her satıra source_group (valve1/valve2) ve source_file (dosya adı) kolonu ekle. 
     Hepsini concat et. Hedef: 'anomaly' (0/1). 
     Sensör kolonları = 8 sensör (datetime, anomaly, changepoint, source_group, source_file HARİÇ).
   - load_batadal(): BATADAL_dataset04.csv oku. TÜM sütun isimlerine .str.strip() uygula. 
     Etiket: 'ATT_FLAG' → map: -999→0, 1→1 (binary). DATETIME'ı sırala/koru ama girdiye katma. 
     Sensör kolonları = 43 değişken.

2. preprocess.py:
   - fit_transform mantığı sklearn tarzı: scaler.fit(X_train) → transform(val/test). 
     PCA da aynı: yalnız train'de fit, n_components config'den (PC1 için 1).
   - Gaussian noise ekleme fonksiyonu (config'deki std ile), sadece test senaryosunda kullanılacak.
   - make_windows(): last-step etiketleme. Y_i = pencere son satırının etiketi. 
     window_size ve stride config'den. Çıktı: (N, window, features) ve (N,) etiket.

3. splits.py:
   - skab_split(): GroupKFold (mümkünse StratifiedGroupKFold), gruplar = source_file. 
     Aynı csv hem train hem test'te olamaz.
   - batadal_split(): zaman sıralı %60/%20/%20 (train/val/test). Rastgele bölme YOK.

LEAKAGE KURALLARI (ZORUNLU):
- Scaler, PCA yalnız train fit. Val/test sadece transform.
- Bu kuralın testini de yaz (basit assert: test verisi scaler'ı fit etmemeli).

ÇIKTI: Her fonksiyon için kısa bir smoke test çalıştır: SKAB concat satır sayısı, 
BATADAL şekil, bir fold örneği, bir window batch şekli. Sonuçları raporla.
```

**Doğrulama kriteri:** SKAB concat ~22.000 satır civarı (20 dosya × ~1100), BATADAL 4177 satır. Window batch şekli (N, window_size, n_features) doğru. GroupKFold'da hiçbir source_file iki kümede yok. Leakage testi geçti.

---

### FAZ 2 — Derin Öğrenme Modelleri (Gün 2-3)
**Sorumlu:** Emirhan. **Tahmini süre:** 1 gün.

**Ne yapılacak:** LSTM, GRU, 1D-CNN — üçü de (rubrik "en az iki" diyor ama üçü de yapılırsa karşılaştırma zengin olur). Many-to-One, last-step label. Sigmoid çıkış, BCE loss, class imbalance için pos_weight (BATADAL). Early stopping (val loss, patience=5), epoch=50, batch=32. Tümü config'den.

#### Antigravity Promptu — DL modelleri
```
GÖREV: src/models/deep/ altında LSTM, GRU ve 1D-CNN sınıflandırıcılarını yaz (PyTorch). 
Tümü config-driven, hard-code yok.

ORTAK FORMÜLASYON:
- Girdi: (batch, window_size, n_features). Çıkış: tek skaler (anomali olasılığı), sigmoid.
- Many-to-One: LSTM/GRU son hidden state → Dense → sigmoid. 
  1D-CNN: Conv1d katmanları → Global Average Pooling → Dense → sigmoid.
- Loss: BCEWithLogitsLoss. Class imbalance için pos_weight parametresi (BATADAL'da ~%5 anomali).
- Eğitim: epochs=50, batch_size=32, early stopping (val loss, patience=5) — hepsi config'den.
- Seed config'den set edilsin (tekrarlanabilirlik).

DOSYALAR:
- lstm.py, gru.py, cnn1d.py: model sınıfları.
- trainer.py: ortak eğitim döngüsü (train/val/early stopping/checkpoint), metrik loglama.

ÇIKTI: Küçük bir sanity-check: rastgele/küçük veriyle her model bir epoch eğitilsin, 
loss düşüyor mu, çıkış şekli doğru mu raporla. GERÇEK eğitimi henüz tam yapma, sadece pipeline çalışsın.
```

**Doğrulama kriteri:** Üç model de forward/backward çalışıyor, loss düşüyor, çıkış (batch,1). Early stopping tetikleniyor. pos_weight uygulanıyor.

---

### FAZ 3 — Otomata Modeli (Gün 3-4)
**Sorumlu:** Efekan. **Tahmini süre:** 1.5 gün (en yoğun faz).

**Ne yapılacak:** PC1 sinyali → PAA → SAX (pyts, quantile, yalnız train fit) → sliding window pattern dizisi → her benzersiz pattern bir state → Add-k smoothing'li geçiş olasılığı matrisi → log-space path probability + uzunluk normalizasyonu → percentile eşiği ile anomali kararı → last-step hizalama.

**Kritik:** SAX sözlüğü ve geçiş olasılıkları YALNIZ train'den. Add-k ile sıfır olasılık engellenir. Path prob log-space, sonra pencere uzunluğuna bölünür (per-transition avg). Güven skoru = bu normalize olasılıktan türer.

#### Antigravity Promptu — Otomata modeli
```
GÖREV: src/models/automata/ altında olasılıksal otomata modelini yaz. config-driven.

PIPELINE (yalnız train'den öğren, test'e uygula — leakage yok):
1. paa_sax.py (pyts kullan):
   - PiecewiseAggregateApproximation + SymbolicAggregateApproximation(strategy='quantile').
   - fit YALNIZ train PC1 sinyalinde. transform test'e uygulanır (donmuş breakpoint).
   - alphabet_size config'den.

2. automata.py:
   - SAX sembol dizisi üzerinde sliding window (window_size config'den) → her pencere bir "pattern".
   - Her benzersiz pattern bir state. State'ler arası geçiş frekansları sayılır.
   - Geçiş olasılığı = frekans tabanlı + Add-k (Lidstone) smoothing. 
     a_ij = (C(i->j) + k) / (sum_k C(i->k) + k*M). k config'den (smoothing_k).
   - Path probability LOG-SPACE'te hesaplanır (underflow engelle): log toplamı.
   - Uzunluk normalizasyonu: toplam log-prob / pencere uzunluğu (per-transition avg log-likelihood).
   - Güven skoru = normalize path probability'den türetilir.
   - Anomali kararı: normalize path-prob, TRAIN dağılımının percentile eşiğinin altındaysa anomali.
     Eşik percentile'ı config'den (örn 5. percentile).
   - Satır kararı: last-step hizalama (DL ile simetri).

ÇIKTI: Küçük sentetik bir PC1 sinyaliyle uçtan uca test: SAX dizisi, state sayısı, 
geçiş matrisi (smoothing öncesi/sonrası sıfır var mı), bir pencerenin log path-prob'u ve güven skoru. Raporla.
```

**Doğrulama kriteri:** SAX dizisi üretiliyor, state sayısı mantıklı, geçiş matrisinde Add-k sonrası sıfır YOK, path prob log-space'te hesaplanıyor ve underflow yok, güven skoru [0,1] aralığında yorumlanabilir.

---

### FAZ 4 — Unseen Pattern Yönetimi + Birim Testler (Gün 4)
**Sorumlu:** Efekan. **Tahmini süre:** yarım gün.

**Ne yapılacak:** Test sırasında train SAX sözlüğünde olmayan pattern = unseen. Levenshtein (edit distance) ile en yakın bilinen pattern bulunur, sistem o state üzerinden devam eder. **Bu mekanizmanın pytest birim testleri ZORUNLU** (rubrik 5 puan).

#### Antigravity Promptu — Unseen + testler
```
GÖREV: src/models/automata/unseen.py ve tests/ altında unseen pattern yönetimi + birim testleri yaz.

1. unseen.py:
   - Train'den SAX pattern sözlüğü çıkar.
   - Test sırasında sözlükte olmayan pattern = unseen olarak işaretle.
   - Levenshtein (edit distance) ile en yakın bilinen pattern'ı bul (python-Levenshtein veya elle).
   - Sistem en yakın pattern'ın state'i üzerinden devam etsin.
   - Unseen oranı (detection rate) ve mapping accuracy raporlanabilsin.

2. tests/test_unseen.py (pytest):
   - Bilinen pattern → unseen DEĞİL.
   - Tamamen yeni pattern → unseen olarak işaretlenir.
   - "adc" gibi bir unseen, "abc" gibi distance=1 olan en yakına map'lenir (rapordaki örnek).
   - Edit distance hesabı doğru mu (birkaç bilinen çift ile).
   - Boş/edge case'ler.

ÇIKTI: pytest çalıştır, tüm testler geçsin. Test çıktısını ve örnek bir unseen→mapping sonucunu göster.
```

**Doğrulama kriteri:** `pytest tests/` tüm testler PASS. Unseen örneği doğru en yakın pattern'a map'leniyor (örn. distance=1).

---

### FAZ 5 — Açıklanabilirlik Modülü (Gün 4-5)
**Sorumlu:** Efekan. **Tahmini süre:** yarım gün.

**Ne yapılacak:** Her karar için: mevcut state, gözlemlenen pattern, train'de var mı, unseen ise mapping, gerçekleşen state geçişleri, her geçişin olasılığı, path probability, güven skoru, nihai karar + olasılıksal gerekçe. JSON + tablo formatı (rapordaki örnek format).

#### Antigravity Promptu — Açıklanabilirlik
```
GÖREV: src/explain/ altında olasılıksal açıklanabilirlik modülünü yaz.

Her karar için şunları üreten bir explain(window) fonksiyonu:
- time_step, current state, gözlemlenen pattern
- pattern train'de var mı (seen/unseen)
- unseen ise: nearest pattern + distance (Levenshtein)
- gerçekleşen state geçişleri ve her birinin olasılığı
- path probability (log ve normalize) 
- güven skoru (confidence) + yorum (low/high → anomali/normal)
- nihai karar (anomaly/normal)

ÇIKTI FORMATI (ikisi de):
1. JSON: {"time_step":5,"state":"aab","pattern":"adc","status":"unseen",
   "mapped_to":"abc","probability":0.108,"decision":"anomaly"}
2. İnsan-okur tablo/metin (rapordaki [SYSTEM DECISION] formatı gibi).

Açıklamalar DETERMİNİSTİK ve modelin iç hesaplamalarıyla TUTARLI olmalı 
(aynı girdi → aynı açıklama, ve sayılar otomata matrisiyle birebir uyumlu).

ÇIKTI: Bir seen ve bir unseen örnek için tam açıklama çıktısı (JSON + tablo) göster.
```

**Doğrulama kriteri:** Seen ve unseen için JSON + tablo üretiliyor, sayılar otomata modelinin matrisiyle tutarlı, deterministik (tekrar çalıştırınca aynı).

---

### FAZ 6 — Deney Protokolü: Senaryolar, Seed'ler, Parametre Analizi (Gün 5)
**Sorumlu:** Ortak. **Tahmini süre:** 1 gün (eğitim süreleri dahil).

**Ne yapılacak:** Tüm modeller × 3 senaryo (original, gaussian noise, unseen) × 5 seed [42,123,2026,7,999]. SKAB GroupKFold fold ortalaması+std, BATADAL zaman sıralı test. Automata parametre sweep (window 3-6, alphabet 3-6). Metrikler: Accuracy, Precision, Recall, F1. Runtime (training+inference) ölçümü. Her run otomatik loglanır.

#### Antigravity Promptu — Deney runner
```
GÖREV: src/experiments/ altında tüm deneyleri yürüten config-driven bir runner yaz.

DENEY MATRİSİ:
- Modeller: LSTM, GRU, 1D-CNN, Automata.
- Veri setleri: SKAB, BATADAL.
- Senaryolar: original, gaussian_noise, unseen (config'den).
- Seed'ler: [42, 123, 2026, 7, 999] — her deney 5 kez.
- SKAB: GroupKFold, fold ortalaması ± std raporla. BATADAL: zaman sıralı test.
- Automata parametre sweep: window_size [3,4,5,6], alphabet_size [3,4,5,6] (F1 etkisi).

METRİKLER: Accuracy, Precision, Recall, F1 (her senaryo/seed/fold için).
RUNTIME: her modelin training time ve inference time (saniye) ölç ve logla.

LOGLAMA:
- Her run'ın parametreleri + metrikleri results/ altına JSON/CSV olarak kaydet.
- Sonuçlar karşılaştırılabilir formatta (model × dataset × senaryo × metrik) toplansın.
- Ortalama ± std otomatik hesaplansın.

ÇIKTI: Önce KÜÇÜK bir alt-küme ile (1 seed, 1 senaryo) tüm pipeline'ın uçtan uca çalıştığını doğrula, 
sonra tam matrisi çalıştır. Sonuç tablolarını (EK belgedeki Tablo 1-5 formatına uygun) üret.
```

**Doğrulama kriteri:** EK belgedeki Tablo 1-5 dolduruluyor (F1±std, gürültü/unseen, cross-dataset, parametre sweep, runtime). Tüm seed/senaryo kombinasyonları loglanmış.

---

### FAZ 7 — İstatistiksel Analiz (Gün 5-6) — ZORUNLU (5 puan)
**Sorumlu:** Ortak. **Tahmini süre:** 2-3 saat.

**Ne yapılacak:** Model farklarının anlamlılığı: Wilcoxon signed-rank (seed/fold bazlı F1 dağılımları) ve/veya McNemar (tahmin bazlı). Hangi model farkı istatistiksel anlamlı, p-değerleriyle tartış.

#### Antigravity Promptu — İstatistiksel testler
```
GÖREV: src/experiments/stats.py altında istatistiksel anlamlılık testlerini yaz (scipy).

- Wilcoxon signed-rank test: model çiftleri arasında (örn LSTM vs Automata) 
  seed/fold bazlı F1 skorları üzerinden. p-değeri raporla.
- McNemar test: model çiftlerinin tahminleri üzerinden (doğru/yanlış kontenjans tablosu).
- Her model çifti için p-değeri tablosu üret, anlamlılık eşiği 0.05.
- Sonuçları yorumla: hangi fark anlamlı, hangisi değil.

ÇIKTI: Model çiftleri × p-değeri tablosu + kısa yorum. SKAB (fold bazlı) ve BATADAL ayrı.
```

**Doğrulama kriteri:** Wilcoxon + McNemar p-değerleri hesaplanmış, model çiftleri tablosu var, anlamlılık yorumlanmış.

---

### FAZ 8 — Görselleştirme + Rapor (Gün 6)
**Sorumlu:** Ortak. **Tahmini süre:** 1 gün.

**Ne yapılacak (görseller, rubrik zorunlu):** Confusion Matrix, ROC veya PR eğrisi, Automata state diagram, Transition probability heatmap, Parametre duyarlılık grafikleri. **Rapor README.md'de Markdown** (rubrik: GitHub readme.md).

**Rapor içeriği (rubrik):** Model karşılaştırmaları, veri setleri arası performans farkları, gürültü etkisi, unseen davranışı, parametre etkileri. Olasılıksal sonuçların (low/high likelihood) yorumu. Akademik yazım.

#### Antigravity Promptu — Görselleştirme
```
GÖREV: src/eval/visualize.py altında tüm rapor görsellerini üreten fonksiyonlar yaz.

GÖRSELLER (results/figures/ altına kaydet):
1. Confusion Matrix (her model/dataset).
2. ROC eğrisi VEYA Precision-Recall eğrisi (imbalance nedeniyle PR önerilir, ikisi de olabilir).
3. Automata state diagram (networkx/graphviz): state'ler düğüm, geçişler olasılık-ağırlıklı kenar.
4. Transition probability heatmap (geçiş matrisi, seaborn).
5. Parametre duyarlılık grafikleri (window_size ve alphabet_size vs F1).

ÇIKTI: Tüm figürleri üret, results/figures/ altına kaydet, dosya listesini göster.
```

#### Antigravity Promptu — README/rapor
```
GÖREV: README.md (proje raporu) yaz — Markdown, akademik ton.

BÖLÜMLER:
- Proje tanımı, araştırma problemi
- Veri setleri (SKAB, BATADAL) ve ön işleme (normalizasyon, PCA, leakage önleme)
- Modeller (LSTM/GRU/1D-CNN ve otomata: PAA/SAX/sliding window/Add-k/path prob)
- Unseen yönetimi (Levenshtein) + birim test özeti
- Açıklanabilirlik modülü (örnek JSON/tablo çıktısı)
- Deneysel tasarım (3 senaryo, 5 seed, split stratejileri)
- Sonuçlar: tüm tablolar (F1±std, gürültü, cross-dataset, parametre, runtime) + figürler
- İstatistiksel analiz (Wilcoxon/McNemar p-değerleri ve yorum)
- Karşılaştırmalı tartışma: model davranışları, veri setine bağımlılık, gürültü etkisi, 
  unseen davranışı, parametre etkileri, olasılıksal yorum (low/high likelihood)
- Sonuç (tek "en iyi model" değil; sistematik analiz vurgusu)

KISIT: Görselleri results/figures/'tan referansla. Tabloları EK belge formatına uygun doldur. 
Akademik yazım, kaynak kullanımı.

ÇIKTI: README.md'yi oluştur ve göster.
```

**Doğrulama kriteri:** 5 görsel türü de üretildi, README tüm bölümleri + tabloları + figürleri içeriyor, akademik ton. EK belgedeki tablolar dolu.

---

## 4. Zaman Çizelgesi (6 Gün)

| Gün | Tarih | Fazlar |
|---|---|---|
| 1 | 1 Haz | Faz 0 (iskelet/config/gitignore) + Faz 1 başlangıç (veri) |
| 2 | 2 Haz | Faz 1 bitiş + Faz 2 (DL modelleri) |
| 3 | 3 Haz | Faz 2 bitiş + Faz 3 (otomata) başlangıç |
| 4 | 4 Haz | Faz 3 bitiş + Faz 4 (unseen/test) + Faz 5 (açıklanabilirlik) |
| 5 | 5 Haz | Faz 6 (deneyler/seed döngüsü) + Faz 7 (istatistik) |
| 6 | 6 Haz | Faz 8 (görsel + rapor) + tampon/düzeltme |
| — | 7 Haz | **Son teslim 23:59** — sadece son kontrol, commit temizliği |

> Tampon az. Faz 6 (tüm seed × senaryo eğitimi) en uzun sürebilen faz — DL eğitim süreleri burada birikir. Mümkünse Faz 6'yı Gün 5 sabahı başlat.

---

## 5. Commit Disiplini (Rubrik — 0 puan riski!)

- **İkiniz de düzenli ve orantılı commit atın.** Hoca commit dağılımına bakıyor; tek kişi commit atarsa proje 0 alabilir.
- Her faz/alt-görev sonunda anlamlı commit (örn. "feat: SKAB loader + leakage-free preprocess").
- Emirhan veri+DL'e, Efekan otomata+açıklanabilirlik'e commit atsın; ortak fazlarda ikiniz de.
- Küçük ve sık commit > tek dev commit.

---

## 6. Ek Puan (Opsiyonel — Çekirdek 100 bitince)

Rubrikte "ek puan kapsamında" geçen, 100'ün dışındaki analizler. Vakit kalırsa:
- **Mean-Aggregation:** overlap'li pencerelerde satır skoru = pencere skorlarının ortalaması (point-wise daha hassas).
- **EVT/POT eşikleme:** percentile yerine Genelleştirilmiş Pareto ile dinamik eşik.
- **Counterfactual analiz:** alternatif pattern'lar altında kararın nasıl değişeceği.
- **Benzerlik tabanlı açıklama:** unseen pattern'ların en yakın pattern'lara mesafelerinin detaylı raporu.

---

## 7. Sıradaki Adım

1. **Şimdi:** Faz 0 promptlarını (.gitignore + iskelet/config) Antigravity'e at.
2. Plan gelince bana göster → onaylayalım → yaptır.
3. Faz 0 bitince Faz 1'e geç. Her fazda bu belgeden ilgili promptu kullan.
4. Takıldığın/karar gereken her noktada bana sor — belge canlı, güncelleriz.
