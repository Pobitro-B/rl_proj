# trainers/dmcontrol_trainer.py
import torch
from torch.optim import Adam
import numpy as np
from collections import deque
from trainers.base_trainer import BaseTrainer

class DMControlPMPOTrainer(BaseTrainer):
    def __init__(self, env, policy, ref_policy=None, pref_model=None, device='cpu',
                 gamma=0.99, K=8, traj_len=200, lambda_div=0.01, lr=3e-4):
        super().__init__(policy, ref_policy, pref_model, device)
        self.env = env
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.K = K
        self.traj_len = traj_len
        self.lambda_div = lambda_div
        self.gamma = gamma

    def rollout(self, init_obs=None):
        obs = self.env.reset() if init_obs is None else init_obs
        traj = []
        for t in range(self.traj_len):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            act = self.policy.sample(obs_t).detach().cpu().numpy()[0]
            next_obs, rew, done, info = self.env.step(act)
            traj.append((obs, act, rew, next_obs, done))
            obs = next_obs
            if done:
                break
        return traj

    def trajectory_score(self, traj):
        # if pref_model: score trajectory embedding; otherwise use discounted return
        if self.pref_model:
            # compute embedding (e.g., mean obs + mean actions)
            obs_feats = np.array([step[0].flatten() for step in traj])
            act_feats = np.array([step[1] for step in traj])
            feat = np.concatenate([obs_feats.mean(axis=0), act_feats.mean(axis=0)], axis=-1)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            return float(self.pref_model(feat_t).item())
        else:
            # discounted return
            G = 0.0; pw = 1.0
            for (_,_,r,_,_) in traj:
                G += pw * r
                pw *= self.gamma
            return float(G)

    def logprob_trajectory(self, traj):
        # approximate logprob of the trajectory under policy as sum_t logpi(a_t|s_t)
        total = 0.0
        for (s,a,_,_,_) in traj:
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device)
            lp = self.policy.log_prob(s_t, a_t).sum()
            total += lp
        return total

    def train_step(self):
        # sample K trajectories from current policy
        trajs = [self.rollout() for _ in range(self.K)]
        scores = [self.trajectory_score(traj) for traj in trajs]
        # build q weights
        q_logits = torch.tensor(scores, dtype=torch.float32)
        q_weights = torch.softmax(self.gamma * q_logits, dim=0).to(self.device)
        # compute expected logp and divergence
        expected_logp = 0.0
        div = 0.0
        for i, traj in enumerate(trajs):
            logp = self.logprob_trajectory(traj).to(self.device)
            expected_logp = expected_logp + q_weights[i] * logp
            if self.ref_policy:
                # approximate ref logp by calling ref policy
                ref_logp = self._ref_logprob_trajectory(traj).to(self.device)
                div = div + q_weights[i] * (logp - ref_logp)
        loss = - expected_logp + self.lambda_div * div
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
