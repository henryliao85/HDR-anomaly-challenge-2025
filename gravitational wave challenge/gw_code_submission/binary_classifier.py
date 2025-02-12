import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_fft import torch_fft

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # 对第一个隐藏层输出加dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # 对第二个隐藏层输出加dropout
            nn.Linear(hidden_dim, 1)  # 最终输出1 => 二分类可再做sigmoid
        )

    def forward(self, x):
        # x shape=(batch, in_dim)
        x = torch_fft(x)
        mu_X = torch.mean(x)
        std_X = torch.std(x)
        logit = self.net((x-mu_X)/std_X)  # shape=(batch,1)
        return logit
