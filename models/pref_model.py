# models/pref_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComparatorNet(nn.Module):
    """
    Score a trajectory or (state, action) pair.
    Input: flattened feature vector (state embedding or pooled CNN features)
    Output: scalar score
    """
    def __init__(self, input_dim, hidden=[256,128]):
        super().__init__()
        layers = []
        inp = input_dim
        for h in hidden:
            layers.append(nn.Linear(inp,h)); layers.append(nn.ReLU()); inp = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(inp, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)
