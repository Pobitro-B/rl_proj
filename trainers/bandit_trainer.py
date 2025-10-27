# trainers/bandit_trainer.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from trainers.base_trainer import BaseTrainer

class BanditPMPOTrainer(BaseTrainer):
    def __init__(self, env, policy, ref_policy=None, pref_model=None, device='cpu', lr=1e-3, gamma=5.0, lambda_div=0.01):
        super().__init__(policy, ref_policy, pref_model, device)
        self.env = env
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_div = lambda_div

    def sample_candidates(self, context, K=8):
        # returns K actions and their log probs under current policy
        obs = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_batch = obs.repeat(K,1)
        with torch.no_grad():
            # for continuous action: sample K actions deterministically via policy.sample
            acts = self.policy.sample(obs_batch)
        acts = acts.cpu().numpy()
        return acts

    def score_actions(self, context, acts):
        # if pref_model available: score via it; else use true reward function
        if self.pref_model:
            obs = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
            obs_batch = obs.repeat(len(acts),1)
            act_tensor = torch.tensor(acts, dtype=torch.float32).to(self.device)
            inp = torch.cat([obs_batch, act_tensor], dim=-1)
            scores = self.pref_model(inp).detach().cpu().numpy()
        else:
            # fallback: query env reward function per action
            scores = np.array([self.env.reward_fn(context, a) for a in acts])
        return scores

    def update_policy(self, context, acts, scores):
        K = len(acts)
        obs = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_batch = obs.repeat(K,1)
        acts_t = torch.tensor(acts, dtype=torch.float32).to(self.device)
        # q over actions
        q_logits = torch.tensor(scores, dtype=torch.float32)
        q_weights = F.softmax(self.gamma * q_logits, dim=0)  # size K
        # log probs under policy
        logp = self.policy.log_prob(obs_batch, acts_t)  # [K]
        logp = logp.squeeze(-1)
        expected_logp = (q_weights.to(self.device) * logp).sum()
        # divergence: use KL between policy and ref at sampled actions (approx)
        if self.ref_policy is not None:
            logp_ref = self.ref_policy.log_prob(obs_batch, acts_t).squeeze(-1).detach()
            div = (q_weights.to(self.device) * (logp - logp_ref)).sum()
        else:
            div = 0.0
        loss = - expected_logp + self.lambda_div * div
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self, K=8):
        ctx = self.env.reset()
        acts = self.sample_candidates(ctx, K=K)
        scores = self.score_actions(ctx, acts)
        loss = self.update_policy(ctx, acts, scores)
        return loss
