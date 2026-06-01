import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, config, n_features):
        super(GRUModel, self).__init__()
        cfg = config['models']['gru']
        self.hidden_size = cfg['hidden_size']
        self.num_layers = cfg['num_layers']
        dropout = cfg.get('dropout', 0.0)
        
        self.gru = nn.GRU(
            input_size=n_features, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True, 
            dropout=dropout if self.num_layers > 1 else 0
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, window_size, n_features)
        out, _ = self.gru(x)
        # out shape: (batch, window_size, hidden_size)
        # Many-to-One: son adımı al
        last_step_out = out[:, -1, :]
        
        logit = self.fc(last_step_out)
        return logit
