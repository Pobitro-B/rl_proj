import torch
import numpy as np

class MPO:
    """
    Maximum a Posteriori Policy Optimization (dual solved)
    """

    def __init__(self, policy, ref_policy, optimizer, epsilon=0.1, eta_init=1.0):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.eta = eta_init

    def e_step(self, rewards):
        q = torch.exp(rewards / self.eta)
        q /= q.sum() + 1e-9
        return q

    def m_step(self, obs, acts, q_weights):
        logp = self.policy.log_prob(obs, acts)
        with torch.no_grad():
            logp_ref = self.ref_policy.log_prob(obs, acts)
        kl = torch.mean(torch.exp(logp_ref) * (logp_ref - logp))
        loss = -torch.sum(q_weights * logp) + self.epsilon * kl
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "kl": kl.item()}

    def train_step(self, obs, acts, rewards):
        q = self.e_step(rewards)
        return self.m_step(obs, acts, q)
