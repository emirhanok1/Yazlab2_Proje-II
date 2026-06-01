**YazLab 2. Proje Rapor Örneği** Deney Sonuçları ve Karşılaştırmalı Analiz Tabloları 

xx. Grup 

04.04.2026 

## **Giriş** 

Bu tamamlayıcı doküman, _"From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis"_ başlıklı ana projenin raporunda yer alması gereken kapsamlı deney sonuçlarını ve detaylı tablo dökümlerini içermektedir. 

## **1 Temel Performans ve Stabilite** 

Aşağıdaki tablo, modellerin iki farklı veri seti üzerindeki ortalama F1-skorlarını ve 5 farklı random seed ile elde edilen standart sapma değerlerini göstermektedir. 

Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score _±_ Standart Sapma) 

|**Model**|**SWAT**|**WADI**|**BATADAL**|
|---|---|---|---|
|LSTM|_±_|_±_|_±_|
|GRU|_±_|_±_|_±_|
|1D-CNN|_±_|_±_|_±_|
|Automata|_±_|_±_|_±_|



## **2 Gürültü ve Unseen Veri Analizi (Robustness)** 

Modellerin veri kalitesindeki düşüşlere ve daha önce karşılaşılmamış örüntülere (unseen patterns) karşı ne kadar dirençli olduğunu ölçmek için Gaussian gürültü eklenmiş veri seti ve görülmemiş veri senaryosu test edilmiştir. 

Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi 

|**Model**|**Gürültü Etkisi (F1)**<br>Orijinal<br>Gürültülü|**Unseen Analizi**<br>Det. Rate<br>Map. Acc.|
|---|---|---|
|LSTM<br>GRU<br>1D-CNN<br>Automata|||



1 

## **3 Çapraz Veri Seti (Cross-Dataset) Genellenebilirliği** 

Bu bölümde modellerin bir veri setinde eğitilip diğerlerinde test edilmesiyle elde edilen genellenebilirlik matrisi sunulmaktadır. 

Tablo 3: Cross-Dataset Performans Karşılaştırması 

|**Train **|**/ Test**|**SWAT**|**WADI**|**BATADAL**|
|---|---|---|---|---|
|Train:|SWAT||||
|Train:|WADI||||
|Train:|BATADAL||||



## **4 Automata Parametre ve Süre Analizi** 

Otomata modelinin iç parametrelerinin (Window Size ve Alphabet Size) performans üzerindeki etkisi ile tüm modellerin eğitim/çıkarım (inference) süreleri aşağıda listelenmiştir. 

Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score) 

|**Parametre**|**Değer= 3**|**Değer= 4**|**Değer= 5**|**Değer= 6**|
|---|---|---|---|---|
|Window Size|||||
|Alphabet Size|||||



Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması 

|**Model**|**Training **|**Time **|**(sn)**|**Inference **|**Time **|**(sn)**|
|---|---|---|---|---|---|---|
|LSTM|||||||
|GRU|||||||
|1D-CNN|||||||
|Automata|||||||



2 

