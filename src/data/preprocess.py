import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fit_scaler(X_train, config):
    """
    StandardScaler'ı yalnızca X_train üzerinde eğitir ve scaler nesnesini döner.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def apply_scaler(X, scaler):
    """
    Önceden eğitilmiş scaler'ı kullanarak veriyi dönüştürür.
    """
    return scaler.transform(X)

def fit_pca(X_train, config):
    """
    PCA'i yalnızca X_train üzerinde eğitir ve pca nesnesini döner.
    Otomata için kullanılacaktır.
    """
    n_components = config['preprocessing']['pca_components']
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return pca

def apply_pca(X, pca):
    """
    Önceden eğitilmiş PCA'i kullanarak veriyi dönüştürür. PC1 sinyalini döner.
    Çıktı (N, 1) veya duruma göre (N,) olabilir, biz 1D numpy array döneceğiz.
    """
    res = pca.transform(X)
    if res.shape[1] == 1:
        return res.flatten()
    return res

def add_gaussian_noise(X, config):
    """
    Verilen X dizisine config['noise']['std'] standart sapmasıyla Gauss gürültüsü ekler.
    Genellikle sadece test verisinde 'gaussian_noise' senaryosu için çağrılır.
    """
    std = config['noise']['std']
    noise = np.random.normal(0, std, X.shape)
    return X + noise

def make_windows(signal_or_X, y, config):
    """
    Veriyi (1D PC1 sinyali veya çok boyutlu X_sensor) sliding window formatına çevirir.
    DL için: (N, window_size, n_features)
    Automata için (tek boyutlu): (N, window_size)
    Etiket stratejisi last-step'tir (pencerenin son satırının etiketi Y_i olur).
    """
    window_size = config['automata']['window_size'] # ya da DL için aynı kullanıyoruz
    stride = config['windowing']['stride']
    
    if isinstance(signal_or_X, list):
        signal_or_X = np.array(signal_or_X)
    if isinstance(y, list):
        y = np.array(y)
        
    X_windows = []
    y_windows = []
    
    n_samples = len(signal_or_X)
    
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        X_windows.append(signal_or_X[start:end])
        y_windows.append(y[end - 1]) # Last-step kuralı
        
    return np.array(X_windows), np.array(y_windows)
