def log_metrics(epoch, train_loss, val_loss, val_f1, train_time=0.0):
    """
    Eğitim döngüsü esnasında loss ve metrikleri ekrana standart formatta basar.
    """
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {train_time:.1f}ms")

def info(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")
