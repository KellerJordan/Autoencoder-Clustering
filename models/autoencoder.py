import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    
    def __init__(self, hidden_act=nn.ReLU(), hidden_units=100):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_units)
        self.linear2 = nn.Linear(hidden_units, 784)
        self.hidden_act = hidden_act
        
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        if self.hidden_act:
            x = self.hidden_act(x)
        x = F.sigmoid(self.linear2(x))
        x = x.view(-1, 1, 28, 28)
        return x

class NonlinearAutoencoder(nn.Module):
    
    def __init__(self, hidden_units, reshape=True):
        super().__init__()
        self.linear1 = nn.Linear(784, 100)
        self.linear2 = nn.Linear(100, hidden_units)
        self.linear3 = nn.Linear(hidden_units, 100)
        self.linear4 = nn.Linear(100, 784)
        self.reshape = reshape
    
    def forward(self, x):
        if self.reshape:
            x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.sigmoid(self.linear4(x))
        if self.reshape:
            x = x.view(-1, 1, 28, 28)
        return x
    