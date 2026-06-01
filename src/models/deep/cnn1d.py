import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self, config, n_features):
        super(CNN1DModel, self).__init__()
        cfg = config['models']['cnn1d']
        channels = cfg['channels']
        kernel_size = cfg['kernel_size']
        dropout = cfg.get('dropout', 0.0)
        
        layers = []
        in_c = n_features
        for out_c in channels:
            layers.append(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_c = out_c
            
        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        # x shape: (batch, window_size, n_features)
        # Conv1D için kanallar (n_features) 1. boyuta gelmeli: (batch, channels, length)
        x = x.permute(0, 2, 1)
        
        feat = self.feature_extractor(x)
        # feat shape: (batch, last_channel, window_size)
        
        pooled = self.global_pool(feat)
        # pooled shape: (batch, last_channel, 1)
        pooled = pooled.squeeze(-1)
        # pooled shape: (batch, last_channel)
        
        logit = self.fc(pooled)
        # logit shape: (batch, 1)
        return logit
