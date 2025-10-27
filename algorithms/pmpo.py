import torch
import torch.nn.functional as F

class PMPO:
    """
    Preference-based MPO (true version)
    Step 1: E-step → compute q(a|s) ∝ exp(r/η)
    Step 2: M-step → update policy π to minimize KL(q || π_ref)
    """

    def __init__(self, policy, ref_policy, optimizer, beta=0.01, eta=1.0, alpha=0.5):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.beta = beta
        self.eta = eta
        self.alpha = alpha
        self.device = next(policy.parameters()).device

    def compute_q_weights(self, rewards):
        """Soft E-step: normalize exp(r/η)"""
        weights = torch.exp(rewards / self.eta)
        weights /= torch.sum(weights) + 1e-9
        return weights

    def train_step(self, obs, acts, rewards):
        """
        obs: [B, obs_dim]
        acts: [B, act_dim]
        rewards: [B]
        """
        logp = self.policy.log_prob(obs, acts)
        with torch.no_grad():
            ref_logp = self.ref_policy.log_prob(obs, acts)

        # E-step: q weights
        w = self.compute_q_weights(rewards)

        # M-step: maximize weighted log π - KL term
        kl = torch.mean(torch.exp(ref_logp) * (ref_logp - logp))
        loss = -torch.sum(w * logp) + self.beta * kl

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "kl": kl.item(), "mean_reward": rewards.mean().item()}
