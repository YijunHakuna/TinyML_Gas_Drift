import torch
import torch.nn as nn

class GasSensorMLP(nn.Module):
    """
    轻量级 MLP，专为端侧部署设计 (参数量估计 < 20k)
    """
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=6):
        super(GasSensorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)