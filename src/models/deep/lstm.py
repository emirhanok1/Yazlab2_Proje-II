import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, config, n_features):
        super(LSTMModel, self).__init__()
        cfg = config['models']['lstm']
        self.hidden_size = cfg['hidden_size']
        self.num_layers = cfg['num_layers']
        dropout = cfg.get('dropout', 0.0)
        
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True, 
            dropout=dropout if self.num_layers > 1 else 0
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, window_size, n_features)
        out, _ = self.lstm(x)
        # out shape: (batch, window_size, hidden_size)
        # Sadece son zaman adımını (last step) alıyoruz
        last_step_out = out[:, -1, :] 
        # last_step_out shape: (batch, hidden_size)
        
        logit = self.fc(last_step_out)
        # logit shape: (batch, 1)
        return logit
