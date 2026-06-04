import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import time
import copy
from src.utils.logging import log_metrics, info

class DeepTrainer:
    def __init__(self, model, config, train_loader, val_loader, pos_weight=None, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Hyperparameters
        train_cfg = config['training']
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['learning_rate']
        self.patience = train_cfg['early_stopping_patience']
        
        # Loss & Optimizer
        if config['models'].get('use_pos_weight', False) and pos_weight is not None:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Memory Best Model (Kullanıcı Talebi: Diske yazma, RAM'de tut)
        self.best_state_dict = None
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        
        self.avg_inference_time_ms = 0.0
        self.total_train_time_ms = 0.0

    def train(self):
        info("Eğitim başlıyor...")
        no_improve_epochs = 0
        total_inf_time = 0.0
        inf_batches = 0
        
        for epoch in range(1, self.epochs + 1):
            # --- TRAIN ---
            self.model.train()
            train_loss = 0.0
            t0 = time.time()
            
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float().unsqueeze(1) # (batch, 1)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_time_ms = (time.time() - t0) * 1000.0
            self.total_train_time_ms += train_time_ms
            train_loss /= len(self.train_loader.dataset)
            
            # --- VALIDATION ---
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device).float()
                    y_batch_real = y_batch.to(self.device).float().unsqueeze(1)
                    
                    t_inf_0 = time.time()
                    logits = self.model(X_batch)
                    t_inf_1 = time.time()
                    total_inf_time += (t_inf_1 - t_inf_0) * 1000.0
                    inf_batches += 1
                    
                    loss = self.criterion(logits, y_batch_real)
                    val_loss += loss.item() * X_batch.size(0)
                    
                    # Logit'leri olasılığa çevir (sigmoid) ve 0.5 eşiğiyle etiketle
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).int()
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(y_batch.numpy().flatten())
                    
            val_loss /= len(self.val_loader.dataset)
            val_f1 = f1_score(all_targets, all_preds, zero_division=0)
            
            log_metrics(epoch, train_loss, val_loss, val_f1, train_time_ms)
            
            # --- EARLY STOPPING & SAVING BEST MODEL (RAM) ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= self.patience:
                info(f"Early stopping tetiklendi! Epoch: {epoch}")
                break
                
        # Eğitim sonu: Best modeli yükle
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            info(f"En iyi model ağırlıkları yüklendi (Epoch {self.best_epoch}).")
            
        if inf_batches > 0:
            self.avg_inference_time_ms = total_inf_time / inf_batches
            
        return self.model
