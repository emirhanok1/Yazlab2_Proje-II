import os
import pandas as pd
import numpy as np

def load_skab(config):
    """
    SKAB (Skoltech Anomaly Benchmark) veri setini config'e göre yükler.
    valve1 ve valve2'deki tüm csv'leri okur, birleştirir.
    Döndürdüğü değerler: X (sensör df), y (anomaly numpy array), groups (source_file serisi).
    """
    skab_cfg = config['datasets']['skab']
    valve1_path = skab_cfg['valve1']
    valve2_path = skab_cfg['valve2']
    sep = skab_cfg['sep']
    decimal = skab_cfg['decimal']
    target_col = skab_cfg['target']
    drop_cols = skab_cfg['drop_from_input']
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    v1_full = os.path.join(base_dir, valve1_path)
    v2_full = os.path.join(base_dir, valve2_path)
    
    dfs = []
    
    # valve1 okuma
    if os.path.exists(v1_full):
        for f in os.listdir(v1_full):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(v1_full, f), sep=sep, decimal=decimal)
                df['source_group'] = 'valve1'
                df['source_file'] = f
                dfs.append(df)
                
    # valve2 okuma
    if os.path.exists(v2_full):
        for f in os.listdir(v2_full):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(v2_full, f), sep=sep, decimal=decimal)
                df['source_group'] = 'valve2'
                df['source_file'] = f
                dfs.append(df)
                
    if not dfs:
        raise FileNotFoundError("SKAB veri dosyaları bulunamadı. Lütfen data/skab klasörünü kontrol edin.")
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # y ayrıştırma
    y = master_df[target_col].astype(float).astype(int).values
    groups = master_df['source_file']
    
    # Girdi (X) ayrıştırma
    cols_to_drop = [c for c in drop_cols if c in master_df.columns]
    X = master_df.drop(columns=cols_to_drop).astype(float)
    
    return X, y, groups


def load_batadal(config):
    """
    BATADAL veri setini config'e göre yükler.
    Sütun boşluklarını temizler, etiket haritalandırır.
    Döndürdüğü değerler: X (sensör df), y (binary numpy array), time (DATETIME serisi).
    """
    batadal_cfg = config['datasets']['batadal']
    path = batadal_cfg['path']
    target_col = batadal_cfg['target']
    target_map = batadal_cfg['target_map']
    time_col = batadal_cfg['time_col']
    strip_names = batadal_cfg['strip_column_names']
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(base_dir, path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"BATADAL veri dosyası bulunamadı: {full_path}")
        
    df = pd.read_csv(full_path)
    
    # Sütun isimlerindeki boşlukları temizleme
    if strip_names:
        df.columns = df.columns.str.strip()
        
    # Y etiket map'leme
    y_raw = df[target_col].copy()
    y = y_raw.map(target_map).values.astype(int)
    
    # Zaman kolonunu ayırma
    time_series = df[time_col].copy()
    
    # X (Girdi) ayırma
    X = df.drop(columns=[target_col, time_col])
    
    return X, y, time_series
