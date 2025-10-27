# models/policy_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicyContinuous(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256], log_std=-0.5):
        super().__init__()
        layers = []
        inp = obs_dim
        for h in hidden:
            layers.append(nn.Linear(inp, h)); layers.append(nn.ReLU()); inp = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(inp, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def log_prob(self, x, actions):
        mu, std = self.forward(x)
        var = std ** 2
        logp = -0.5 * (((actions - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + torch.log(torch.tensor(2 * 3.14159)))
        return logp.sum(dim=-1)
