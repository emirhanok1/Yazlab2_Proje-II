1 

## - Yazılım Geli¸stirme Dersi 2. Proje From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis 

## I. PROJE TANIMI VE MOTIVASYON 

Zaman serisi verileri; finansal sistemler, biyomedikal sinyaller, IoT altyapıları ve davranı¸ssal analiz uygulamaları gibi birçok alanda yaygın olarak kullanılmaktadır. Bu tür veriler üzerinde gerçekle¸stirilen sınıflandırma ve anomali tespiti problemleri, hem akademik ara¸stırmalar hem de endüstriyel uygulamalar açısından kritik öneme sahiptir. 

Bu proje kapsamında, zaman serisi verileri üzerinde iki farklı modelleme paradigmasının kar¸sıla¸stırılması hedeflenmektedir: 

- Derin ö˘grenme tabanlı, yüksek do˘gruluk potansiyeline sahip ancak yorumlanabilirli˘gi sınırlı olan black-box modeller 

- Sembolik temsil ve durum geçi¸slerine dayalı, yorumlanabilir (interpretable) otomata tabanlı modeller 

Proje, modellerin yalnızca performans açısından de˘gil; aynı zamanda genellenebilirlik, gürültüye dayanıklılık ve açıklanabilirlik gibi kriterler çerçevesinde analiz edilmesini amaçlamaktadır. 

## II. ARA ¸STIRMA PROBLEMI VE AMAÇ 

Bu çalı¸smada a¸sa˘gıdaki temel ara¸stırma problemi ele alınmaktadır: 

_Farklı modelleme yakla¸sımları, zaman serisi verileri üzerinde farklı veri ko¸sulları altında nasıl davranmaktadır ve bu davranı¸slar istatistiksel olarak anlamlı mıdır?_ 

Bu kapsamda proje a¸sa˘gıdaki hedefleri içermektedir: 

- Farklı modelleme yakla¸sımlarının kar¸sıla¸stırmalı analizi 

- Model performansının veri setine ba˘gımlılı˘gının incelenmesi 

- Gürültü ve bilinmeyen veri durumlarında model davranı¸sının de˘gerlendirilmesi 

- Açıklanabilirlik açısından modellerin analiz edilmesi 

dosyalarını concat i¸slemi ile birle¸stirerek tek bir veri seti olu¸sturmalıdır. 

Birle¸stirme sırasında a¸sa˘gıdaki ek sütunlar olu¸sturulmalıdır: 

- source_group: Kaydın valve1 veya valve2 klasöründen geldi˘gini gösterir. 

- source_file: Kaydın hangi .csv dosyasından geldi˘gini gösterir. 

Bu ek sütunlar model girdisi olarak kullanılmayacaktır. Yalnızca veri takibi, dosya bazlı veri bölme ve sonuç analizi amacıyla kullanılacaktır. 

SKAB veri setinde hedef de˘gi¸sken anomaly sütunudur. Model girdisi olarak yalnızca sensör de˘gi¸skenleri kullanılmalıdır. datetime, changepoint, source_group ve source_file sütunları model girdisine dahil edilmemelidir. 

## _B. BATADAL Veri Setinin Kullanımı_ 

BATADAL veri seti için yalnızca Training Dataset 2 kullanılacaktır. Ö˘grenciler bu veri setinde yer alan sensör ve sistem de˘gi¸skenlerini model girdisi olarak kullanmalı, saldırı/anomali bilgisini gösteren etiket sütununu ise hedef de˘gi¸sken olarak ele almalıdır. Etiket sütununun adı veri dosyası üzerinde kontrol edilmeli ve raporda açıkça belirtilmelidir. 

Training Dataset 1 yalnızca normal operasyon verisi içerdi˘gi için bu proje kapsamında zorunlu supervised sınıflandırma verisi olarak kullanılmayacaktır. Test Dataset ise etiket bilgisi içermedi˘gi için model performansının de˘gerlendirilmesinde kullanılmayacaktır. 

BATADAL veri setinde zaman bilgisini içeren sütunlar do˘grudan model girdisi olarak kullanılmamalıdır. Bu sütunlar yalnızca zaman sırasının korunması, veri bölme ve sonuçların zamansal olarak yorumlanması amacıyla kullanılmalıdır. 

## IV. VERI ÖN[˙] I ¸SLEME 

Veri ön i¸sleme süreci a¸sa˘gıdaki adımları içermelidir: 

## III. VERI SETI SEÇIMI VE KULLANIMI 

Bu proje kapsamında tüm veri setleri anomali tespiti problemi olarak ele alınacaktır. Her grup iki farklı veri seti üzerinde çalı¸sacaktır: 

## _•_ **SKAB** 

- **BATADAL** 

- Veri normalizasyonu 

- Gerekli durumlarda eksik veri i¸slemleri 

- Çok de˘gi¸skenli veri için boyut indirgeme (PCA) 

Otomata tabanlı model yalnızca tek boyutlu veri ile çalı¸stı˘gı için, çok de˘gi¸skenli veri setlerinde tüm özellikler PCA ile tek boyuta indirgenmeli ve ilk bile¸sen (PC1) kullanılmalıdır. 

## _A. SKAB Veri Setinin Kullanımı_ 

SKAB veri seti için yalnızca valve1 ve valve2 klasörleri kullanılacaktır. Ö˘grenciler bu iki klasörde yer alan tüm .csv 

## V. MODELLEME YAKLA ¸SIMLARI 

## _A. Derin Ö˘grenme Modeli_ 

Ö˘grenciler a¸sa˘gıdaki modellerden en az ikisini uygulamalıdır: 

2 

- LSTM 

- GRU 

- 1D-CNN 

Model e˘gitimi, do˘grulama ve test süreçleri açık bir ¸sekilde raporlanmalıdır. 

- Model performansı 

- State sayısı 

- Geçi¸s yo˘gunlu˘gu 

üzerindeki etkileri analiz edilmelidir. 

## _B. Standart Deney Protokolü (Zorunlu)_ 

## _B. Otomata Tabanlı Model_ 

Otomata modeli a¸sa˘gıdaki dönü¸sümler üzerinden in¸sa edilecektir: 

- PAA (Piecewise Aggregate Approximation) 

- SAX (Symbolic Aggregate approXimation) 

- Sliding Window ile örüntü (pattern) çıkarımı 

Her benzersiz pattern bir durum (state) olarak tanımlanır ve durumlar arası geçi¸s olasılıkları hesaplanır. 

## VI. UNSEEN PATTERN YÖNETIMI 

Test a¸samasında daha önce gözlemlenmemi¸s örüntüler ile kar¸sıla¸sılması durumunda: 

- Levenshtein (Edit Distance) algoritması uygulanmalıdır 

- En yakın pattern belirlenerek sistem bu state üzerinden devam etmelidir 

Bu mekanizmanın birim testlerle do˘grulanması zorunludur. 

Tüm gruplar için deneylerin kar¸sıla¸stırılabilir olması amacıyla veri bölme stratejisi veri setinin yapısına uygun ¸sekilde uygulanmalıdır. Satır bazlı rastgele bölme, zaman serisi ba˘gımlılı˘gını ve deney bütünlü˘günü bozabilece˘gi için kullanılmamalıdır. 

## **SKAB veri seti için:** 

- source_file sütunu grup de˘gi¸skeni olarak kullanılmalıdır. 

- Aynı .csv dosyasına ait kayıtlar hem e˘gitim hem de test kümesinde aynı anda yer almamalıdır. 

- GroupKFold veya mümkünse StratifiedGroupKFold uygulanmalıdır. 

- Sonuçlar fold ortalaması ve standart sapması ile raporlanmalıdır. 

## **BATADAL veri seti için:** 

- Zaman sırası korunmalıdır. 

- Rastgele satır bazlı bölme yapılmamalıdır. 

- Veri zaman sırasına göre %60 e˘gitim, %20 do˘grulama ve %20 test olarak ayrılmalıdır. 

- Farklı oran kullanımı kabul edilmez. 

## _A. Unseen Veri Senaryosunun Tanımlanması_ 

Unseen veri a¸sa˘gıdaki yöntem ile olu¸sturulmalıdır. 

- E˘gitim verisinden elde edilen SAX sözlü˘gü çıkarılır 

- Test sırasında sözlükte bulunmayan pattern’lar unseen olarak kabul edilir 

## **Veri sızıntısı (data leakage) önleme kuralları:** 

- Normalizasyon yalnızca **train** verisi üzerinde fit edilmelidir. 

- Aynı dönü¸süm validation ve test verisine uygulanmalıdır. 

- PCA yalnızca **train** verisi üzerinde fit edilmelidir. 

- SAX/PAA sözlü˘gü ve otomata geçi¸s olasılıkları yalnızca **train** verisi kullanılarak olu¸sturulmalıdır. 

## VII. DENEYSEL TASARIM 

## **Model e˘gitim parametreleri (tüm modeller için sabit):** 

Deneyler üç farklı senaryo altında yürütülmelidir: 

- Orijinal veri 

- Gürültü eklenmi¸s veri (Gaussian noise) 

- Unseen veri 

- Epoch üst sınırı: 50 

- Batch size: 32 

- Early stopping: validation loss (patience= 5) 

- Random seed: 42, 123, 2026, 7, 999 

## VIII. YAZILIM MIMARISI VE TASARIM GEREKSINIMLERI 

## _A. Parametre Analizi_ 

˙Iki a¸samalı bir deney tasarımı uygulanacaktır: 

## **a- Sabit parametreler (kar¸sıla¸stırma için):** 

- window size= 4 

- alphabet size= 3 

## **b- Parametre varyasyonu:** 

- window size: 3, 4, 5, 6 

- alphabet size: 3, 4, 5, 6 

Parametre de˘gi¸simlerinin: 

Proje, parametrik ve modüler bir yazılım mimarisi ile geli¸stirilmelidir. 

A¸sa˘gıdaki gereksinimler zorunludur: 

- Tüm parametreler merkezi bir konfigürasyon yapısında tutulmalıdır 

- Veri i¸sleme ve modelleme süreçleri pipeline yapısı ile tasarlanmalıdır 

- Parametre de˘gi¸siklikleri sistemin tamamını otomatik olarak yeniden olu¸sturmalıdır 

Hard-coded de˘ger kullanımı kabul edilmemektedir. 

3 

## _A. Deney Takibi ve Loglama_ 

- Her deneyin parametreleri kaydedilmelidir 

- Performans metrikleri otomatik loglanmalıdır 

- Nihai karar ve bu kararın olasılıksal gerekçesi 

Bu açıklamalar deterministik, yeniden üretilebilir ve modelin iç hesaplamaları ile tutarlı olmalıdır. 

- Deney sonuçları kar¸sıla¸stırılabilir formatta saklanmalıdır 

## IX. DE GERLENDIRME[˘] METRIKLERI VE[˙] ISTATISTIKSEL ANALIZ 

Model performansı a¸sa˘gıdaki metriklerle de˘gerlendirilmelidir: 

## _B. Güven Skoru_ 

Modelin verdi˘gi her karar için bir **güven skoru** hesaplanmalıdır. Bu skor, otomata üzerindeki geçi¸s olasılıkları kullanılarak elde edilmelidir. 

- Accuracy 

- Precision 

- Recall 

- F1-score 

De˘gerlendirme stratejisi veri setine uygun ¸sekilde uygulanmalıdır: 

- SKAB veri seti için dosya bazlı GroupKFold veya StratifiedGroupKFold kullanılmalıdır. 

- BATADAL veri seti için zaman sıralı %60 e˘gitim, %20 do˘grulama ve %20 test ayrımı kullanılmalıdır. 

- Uygun durumlarda Wilcoxon veya McNemar testi uygulanarak model farklarının istatistiksel anlamlılı˘gı tartı¸sılmalıdır. 

- _A. Deney Tekrarı ve Istatistiksel[˙] Güvenilirlik_ 

- Her deney 5 farklı random seed [ **42, 123, 2026, 7, 999** ] ile çalı¸stırılmalıdır. 

- Sonuçlar ortalama ve standart sapma olarak raporlanmalıdır. 

## _C. Geçi¸s Olasılıklarının Hesaplanması_ 

Olasılıksal otomata modeli, durumlar arası geçi¸s olasılıklarını frekans tabanlı olarak ö˘grenir: 

**==> picture [149 x 23] intentionally omitted <==**

Bir örüntü dizisinin olasılı˘gı, ardı¸sık geçi¸s olasılıklarının çarpımı ile hesaplanır: 

**==> picture [141 x 15] intentionally omitted <==**

Dü¸sük olasılı˘ga sahip diziler, model tarafından beklenmeyen davranı¸slar olarak de˘gerlendirilir ve anomali adayı olarak i¸saretlenir. 

## **Yorumlama:** 

   - Dü¸sük olasılık _→_ Anomali olasılı˘gı yüksek 

   - Yüksek olasılık _→_ Normal davranı¸s 

- SKAB için fold bazlı sonuçlar ayrıca verilmelidir. 

- BATADAL için zaman sıralı test kümesi üzerindeki sonuçlar ayrı raporlanmalıdır. 

## _D. Opsiyonel Geli¸smi¸s Analizler_ 

A¸sa˘gıdaki yöntemler ek puan kapsamında de˘gerlendirilecektir: 

## X. OLASILIKSAL AÇIKLANABILIRLIK MODÜLÜ 

Bu projede geli¸stirilen açıklanabilirlik modülü, do˘grudan **olasılıksal otomata modelinin** iç yapısına dayalı olarak tasarlanmalıdır. 

Amaç, modelin yalnızca tahmin üretmesi de˘gil, aynı zamanda karar sürecinin **olasılıksal geçi¸sler üzerinden matematiksel olarak gerekçelendirilmesidir** . 

- **Benzerlik Tabanlı Açıklama:** Unseen pattern’ların en yakın pattern’lara olan mesafelerinin raporlanması 

- **Kar¸sıt Durum Analizi (Counterfactual):** Alternatif pattern’lar altında model çıktısının nasıl de˘gi¸sece˘ginin analiz edilmesi 

## _E. Örnek Açıklama_ 

[SYSTEM DECISION] 

## _A. Temel Gereksinimler (Zorunlu)_ 

Açıklanabilirlik modülü her karar için a¸sa˘gıdaki bilgileri üretmelidir: 

- Mevcut durum (state) 

- Gözlemlenen örüntü (pattern) 

Time Step: t = 5 Previous State: "aab" 

Incoming Pattern: "adc" Status: Unseen 

- Örüntünün e˘gitim verisinde bulunup bulunmadı˘gı 

- Unseen durumunda uygulanan e¸sleme mekanizması 

Nearest Pattern: "abc" (distance = 1) 

- Gerçekle¸sen durum geçi¸sleri (state transitions) 

- Her geçi¸sin olasılı˘gı 

- Gözlemlenen örüntü dizisinin toplam olasılı˘gı (path probability) 

## Transitions: 

aab -> abc : 0.72 abc -> bcc : 0.15 

4 

Path Probability: 0.72 * 0.15 = 0.108 

Decision: Low probability path detected Result: ANOMALY Confidence Score: 0.108 (Low) 

_F. Çıktı Formatı (Zorunlu)_ 

- Kopya çekti˘gi/intihal yaptı˘gı tespit edilen projeler **0 (sıfır)** olarak notlandırılacaktır. 

- GitHub’a düzenli ve ekip üyesi sayısına göre orantılı commit yapmayanlar **0 (sıfır)** olarak notlandırılacaktır. 

- Belirtilen tarihe kadar grup olu¸sturmayan, ekip arkada¸sı olmasa dahi listeye ismini eklemeyen, proje dosyalarının gönderimini sa˘glamayan ö˘grenciler ve projeler **0 (sıfır)** olarak notlandırılacaktır. 

- Her grup yalnızca tek ve ortak bir proje geli¸stirip sunacaktır. Ayrı ayrı sunum alınmayacak olup ekibin tamamı **0 (sıfır)** olarak notlandırılacaktır. 

Model çıktıları a¸sa˘gıdaki formatlardan biri ile sunulmalıdır: 

**JSON formatı:** 

{ "time_step": 5, "state": "aab", "pattern": "adc", "status": "unseen", "mapped_to": "abc", "probability": 0.108, "decision": "anomaly" } 

veya tablo formatında raporlanmalıdır. 

## XI. RAPORLAMA VE BEKLENTILER 

Rapor a¸sa˘gıdaki analizleri içermelidir. GitHub üzerinde readme.md dosyasında Markdown kullanılarak yazılmalıdır. 

- Model kar¸sıla¸stırmaları 

- Veri setleri arası performans farkları 

- Gürültü etkisi analizi 

- Unseen veri davranı¸sı 

- Parametre etkileri 

Bu proje kapsamında amaç, tek bir en iyi modeli belirlemekten ziyade, model davranı¸slarını bilimsel ve sistematik bir ¸sekilde analiz etmektir. 

Rapor a¸sa˘gıdaki görselleri içermelidir: 

- Confusion Matrix 

- ROC veya Precision-Recall e˘grisi (uygunsa) 

- Automata state diagram 

- Transition probability heatmap 

- Parametre duyarlılık grafikleri 

## XII. TARIHLER 

Proje süreçlerine ait takvim a¸sa˘gıda belirtilmi¸stir: 

**Proje Son Teslim Tarihi:** 7 Haziran 2026 Pazar, Saat 23.59 **Proje Sunum Tarihleri:** 8-12 Haziran 2026 tarihleri arasında yapılacaktır. (Son teslim tarihinden sonra sunum listeleri ilan edilecektir.) 

## **Önemli Notlar ve Ihlal[˙] Durumları:** 

5 

## Tablo I 

## DE GERLENDIRME[˘] KRITERLERI (RUBRIK) 

|**De˘gerlendirme Kriteri**|**Puan**|
|---|---|
|**1. Yazılım Mimarisi ve Kod Kalitesi**|**20**|
|- Merkezi konfgürasyon yapısı ve parametre ba˘gımlı otomatik model üretimi (8 Puan)||
|- Modüler pipeline mimarisi (veri akı¸sı + model entegrasyonu) (7 Puan)||
|- Kod kalitesi, isimlendirme, Git kullanımı ve proje organizasyonu (5 Puan)||
|**2. Veri Ön ˙I¸sleme ve Modelleme Do˘grulu˘gu**|**25**|
|- ˙Iki veri seti için do˘gru ön i¸sleme (normalizasyon, PCA vb.) (5 Puan)||
|- Derin ö˘grenme modelinin do˘gru kurulumu ve e˘gitimi (5 Puan)||
|- Otomata modelinin do˘gru in¸sası (PAA, SAX, sliding window) (5 Puan)||
|- Geçi¸s olasılıklarının do˘gru hesaplanması ve gerekirse smoothing uygulanması (5 Puan)||
|- Unseen veri yönetimi (Levenshtein) ve buna ait birim testler (5 Puan)||
|**3. Olasılıksal Açıklanabilirlik Modülü**|**20**|
|- State, pattern, transition ve unseen mekanizmasının do˘gru raporlanması (5 Puan)||
|- Geçi¸s olasılıklarının açık ¸sekilde sunulması (5 Puan)||
|- Path probability (veya e¸sde˘geri) hesaplanması (5 Puan)||
|- Güven skorunun (confdence) do˘gru tanımlanması ve yorumlanması (5 Puan)||
|**4. Deneysel Tasarım ve ˙Istatistiksel Analiz**|**15**|
|- Farklı senaryoların (normal, gürültü, unseen) sistematik olarak test edilmesi (5 Puan)||
|- Parametre etkisinin (window size, alphabet size) analiz edilmesi (5 Puan)||
|- Veri setine uygun de˘gerlendirme stratejisinin (SKAB için GroupKFold, BATADAL için zaman sıralı de˘gerlendirme) ve||
|istatistiksel testlerin do˘gru uygulanması (5 Puan)||
|**5. Akademik Raporlama ve Analitik Derinlik**|**20**|
|- Veri setleri arası kar¸sıla¸stırmalı analiz ve model davranı¸sı yorumlama (8 Puan)||
|- Olasılıksal sonuçların (low/high likelihood) do˘gru yorumlanması (5 Puan)||
|- Görselle¸stirme (automata grafkleri, performans tabloları) (3 Puan)||
|- Akademik yazım, yapı ve kaynak kullanımı (4 Puan)||
|**TOPLAM**|**100**|



