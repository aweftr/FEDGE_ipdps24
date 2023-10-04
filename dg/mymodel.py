# %%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# %%
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1) -> None:
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.fc_stack(x)
        return x


class QNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    
    def forward(self, x):
        x = self.fc_stack(x)
        return x

class PNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(32, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        x = self.fc_stack(x)
        return x

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc_stack(x)
        return x

class RNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim + 32, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.fc_stack(x)
        return x

class FEDGE(nn.Module):
    def __init__(self, app_dim, noapp_dim):
        super().__init__()
        self.Q = QNet(app_dim)
        self.P = PNet(app_dim)
        self.D = DNet()
        self.R = RNet(noapp_dim)

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(input_dim), requires_grad=True)
        self.sigma = sigma
        self.noise = torch.randn(self.mu.shape).to(device)
    
    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 
    
    def reg_loss(self):
        reg = torch.mean(self.regularizer((self.mu + 0.5) / self.sigma))
        return reg
    
    def get_gates(self, mode="prob"):
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5))
        else:
            raise NotImplementedError()