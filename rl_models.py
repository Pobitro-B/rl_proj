# rl_models.py
"""
Realistic RL wrappers for PMPO, DPO, MPO, IPO.
These classes assume the trainer provides:
- For bandit trainer:
    trainer.env
    trainer.sample_candidates(context, K)
    trainer.score_actions(context, acts)  -> returns array-like scores
    trainer.policy: provides .log_prob(obs, acts) and .sample / .sample_action as implemented
    trainer.ref_policy: same interface (may be None)
    trainer.optimizer
    trainer.device
- For dmcontrol trainer:
    trainer.rollout()
    trainer.trajectory_score(traj)
    trainer.logprob_trajectory(traj)
    trainer._ref_logprob_trajectory(traj)  (optional)
    trainer.policy, trainer.ref_policy, trainer.optimizer, trainer.device

All train(...) methods return a list of logged dicts.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# Utility helpers
# -----------------------------
def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def safe_softmax(x, axis=-1, eps=1e-12):
    x = np.array(x, dtype=float)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    s = e / (e.sum(axis=axis, keepdims=True) + eps)
    return s


# -----------------------------
# PMPO
# -----------------------------
class PMPO_RL:
    """
    Preference MPO (EM-style):
      - E-step: build q over sampled candidates ~ softmax(gamma * scores)
      - M-step: maximize E_q[log pi(y|x)] - lambda_div * divergence proxy
    For trajectories: sample K rollouts and score them (returns or comparator).
    For bandits: sample K actions and score them.
    """

    def __init__(self, trainer, gamma=4.0, lambda_div=0.01, K=8, device="cpu", eval_fn=None):
        self.trainer = trainer
        self.gamma = gamma
        self.lambda_div = lambda_div
        self.K = K
        self.device = device
        self.eval_fn = eval_fn

    def _compute_q_weights(self, scores):
        # scores: numpy array or torch tensor (K,)
        # q ∝ softmax(gamma * scores)
        scores_np = np.array(scores, dtype=float)
        logits = self.gamma * (scores_np - np.max(scores_np))
        q = np.exp(logits)
        q = q / (q.sum() + 1e-12)
        return q  # numpy

    def train(self, num_steps=1000, eval_interval=100):
        logs = []
        for step in range(1, num_steps + 1):
            # Bandit mode (sample candidates per context)
            if hasattr(self.trainer, "sample_candidates"):
                ctx = self.trainer.env.reset()
                acts = self.trainer.sample_candidates(ctx, K=self.K)  # shape: (K, act_dim)
                scores = self.trainer.score_actions(ctx, acts)  # length K
                q = self._compute_q_weights(scores)  # numpy (K,)
                # prepare tensors
                obs = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                obs_batch = obs.repeat(self.K, 1)
                acts_t = torch.tensor(np.array(acts), dtype=torch.float32, device=self.device)
                # log prob under policy
                logp = self.trainer.policy.log_prob(obs_batch, acts_t).squeeze(-1)  # [K]
                # soft reference logp if exists
                if getattr(self.trainer, "ref_policy", None) is not None:
                    with torch.no_grad():
                        logp_ref = self.trainer.ref_policy.log_prob(obs_batch, acts_t).squeeze(-1)
                else:
                    logp_ref = torch.zeros_like(logp).detach()
                # expected logp under q
                q_t = torch.tensor(q, dtype=torch.float32, device=self.device)
                expected_logp = (q_t * logp).sum()
                # divergence proxy: q-weighted (logp - logp_ref)
                div = (q_t * (logp - logp_ref)).sum()
                loss = - expected_logp + self.lambda_div * div
                # step
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item()), "expected_logp": float(expected_logp.item()),
                            "div": float(div.item()), "eval_mean": None}
            else:
                # Trajectory mode (dmcontrol)
                trajs = [self.trainer.rollout() for _ in range(self.K)]
                scores = [self.trainer.trajectory_score(t) for t in trajs]
                q = self._compute_q_weights(scores)
                expected_logp = 0.0
                div = 0.0
                # accumulate as torch scalar
                tot_loss = None
                for i, traj in enumerate(trajs):
                    logp = self.trainer.logprob_trajectory(traj)  # should be torch scalar
                    if isinstance(logp, torch.Tensor):
                        lp = logp.to(self.device)
                    else:
                        lp = torch.tensor(float(logp), dtype=torch.float32, device=self.device)
                    ref_lp = None
                    if getattr(self.trainer, "_ref_logprob_trajectory", None):
                        ref_lp = self.trainer._ref_logprob_trajectory(traj)
                        if not isinstance(ref_lp, torch.Tensor):
                            ref_lp = torch.tensor(float(ref_lp), dtype=torch.float32, device=self.device)
                    else:
                        ref_lp = torch.tensor(0.0, device=self.device)
                    q_i = float(q[i])
                    # compute parts
                    if tot_loss is None:
                        tot_loss = - q_i * lp + self.lambda_div * q_i * (lp - ref_lp)
                    else:
                        tot_loss = tot_loss + (- q_i * lp + self.lambda_div * q_i * (lp - ref_lp))
                loss = tot_loss
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item()), "eval_mean": None}

            if step % eval_interval == 0 or step == 1:
                eval_res = self.eval_fn() if self.eval_fn else {}
                if isinstance(recorded.get("eval_mean", None), type(None)):
                    recorded.update(eval_res)
                print(f"[PMPO] step {step} loss {recorded['loss']:.4f} eval {eval_res}")
                logs.append({"step": step, **recorded, **eval_res})
        return logs


# -----------------------------
# DPO
# -----------------------------
class DPO_RL:
    """
    Direct Preference Optimization for RL:
      - Sample pairs of candidates (actions or trajectories)
      - Use the DPO loss on log-prob differences vs reference
    """

    def __init__(self, trainer, beta=1.0, eval_fn=None):
        self.trainer = trainer
        self.beta = beta
        self.eval_fn = eval_fn
        self.device = trainer.device

    def train(self, num_steps=1000, eval_interval=100):
        logs = []
        for step in range(1, num_steps + 1):
            # Bandit case: sample context, two actions
            if hasattr(self.trainer, "sample_candidates"):
                ctx = self.trainer.env.reset()
                acts = self.trainer.sample_candidates(ctx, K=2)
                scores = self.trainer.score_actions(ctx, acts)
                # decide winner by score (or comparator)
                winner = 0 if scores[0] >= scores[1] else 1
                obs = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                acts_t = torch.tensor(np.array(acts), dtype=torch.float32, device=self.device)
                # compute logps
                logp = self.trainer.policy.log_prob(obs.repeat(2, 1), acts_t).squeeze(-1)
                with torch.no_grad():
                    logp_ref = self.trainer.ref_policy.log_prob(obs.repeat(2, 1), acts_t).squeeze(-1)
                w_idx = winner
                l_idx = 1 - winner
                z = self.beta * ((logp[w_idx] - logp[l_idx]) - (logp_ref[w_idx] - logp_ref[l_idx]))
                loss = F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}
            else:
                # trajectories: sample two rollouts and compare returns
                trajA = self.trainer.rollout()
                trajB = self.trainer.rollout()
                sA = self.trainer.trajectory_score(trajA)
                sB = self.trainer.trajectory_score(trajB)
                winner = "A" if sA >= sB else "B"
                logpA = self.trainer.logprob_trajectory(trajA)
                logpB = self.trainer.logprob_trajectory(trajB)
                if not isinstance(logpA, torch.Tensor):
                    logpA = torch.tensor(float(logpA), device=self.device)
                if not isinstance(logpB, torch.Tensor):
                    logpB = torch.tensor(float(logpB), device=self.device)
                with torch.no_grad():
                    refA = self.trainer._ref_logprob_trajectory(trajA) if getattr(self.trainer, "_ref_logprob_trajectory", None) else torch.tensor(0.0)
                    refB = self.trainer._ref_logprob_trajectory(trajB) if getattr(self.trainer, "_ref_logprob_trajectory", None) else torch.tensor(0.0)
                if not isinstance(refA, torch.Tensor):
                    refA = torch.tensor(float(refA), device=self.device)
                if not isinstance(refB, torch.Tensor):
                    refB = torch.tensor(float(refB), device=self.device)
                if winner == "A":
                    z = self.beta * ((logpA - logpB) - (refA - refB))
                else:
                    z = self.beta * ((logpB - logpA) - (refB - refA))
                loss = F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}

            if step % eval_interval == 0 or step == 1:
                eval_res = self.eval_fn() if self.eval_fn else {}
                print(f"[DPO] step {step} loss {recorded['loss']:.4f} eval {eval_res}")
                logs.append({"step": step, **recorded, **eval_res})

        return logs


# -----------------------------
# MPO
# -----------------------------
class MPO_RL:
    """
    MPO approximate wrapper:
    - E-step: q_i ∝ exp(returns / eta)
    - M-step: gradient-based approx: minimize -sum q_i log π(τ_i) + lambda_kl * divergence
    """

    def __init__(self, trainer, eta=1.0, lambda_kl=0.1, K=8, eval_fn=None):
        self.trainer = trainer
        self.eta = eta
        self.lambda_kl = lambda_kl
        self.K = K
        self.eval_fn = eval_fn
        self.device = trainer.device

    def train(self, num_steps=1000, eval_interval=100):
        logs = []
        for step in range(1, num_steps + 1):
            if hasattr(self.trainer, "rollout"):
                trajs = [self.trainer.rollout() for _ in range(self.K)]
                returns = np.array([self.trainer.trajectory_score(t) for t in trajs], dtype=float)
                # E-step
                logits = returns / (self.eta + 1e-12)
                logits = logits - logits.max()
                q = np.exp(logits); q = q / (q.sum() + 1e-12)
                # M-step approx
                loss = None
                for i, traj in enumerate(trajs):
                    logp = self.trainer.logprob_trajectory(traj)
                    if not isinstance(logp, torch.Tensor):
                        logp = torch.tensor(float(logp), dtype=torch.float32, device=self.device)
                    # ref logp
                    if getattr(self.trainer, "_ref_logprob_trajectory", None):
                        ref_lp = self.trainer._ref_logprob_trajectory(traj)
                        if not isinstance(ref_lp, torch.Tensor):
                            ref_lp = torch.tensor(float(ref_lp), dtype=torch.float32, device=self.device)
                    else:
                        ref_lp = torch.tensor(0.0, device=self.device)
                    q_i = float(q[i])
                    this = - q_i * logp + self.lambda_kl * q_i * (logp - ref_lp)
                    loss = this if loss is None else loss + this
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}
            else:
                # Bandit: single context, K actions
                ctx = self.trainer.env.reset()
                acts = self.trainer.sample_candidates(ctx, K=self.K)
                scores = self.trainer.score_actions(ctx, acts)
                logits = np.array(scores) / (self.eta + 1e-12)
                logits = logits - logits.max()
                q = np.exp(logits); q = q / (q.sum() + 1e-12)
                obs = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                obs_batch = obs.repeat(self.K, 1)
                acts_t = torch.tensor(np.array(acts), dtype=torch.float32, device=self.device)
                logp = self.trainer.policy.log_prob(obs_batch, acts_t).squeeze(-1)
                if self.trainer.ref_policy is not None:
                    with torch.no_grad():
                        logp_ref = self.trainer.ref_policy.log_prob(obs_batch, acts_t).squeeze(-1)
                else:
                    logp_ref = torch.zeros_like(logp).detach()
                q_t = torch.tensor(q, dtype=torch.float32, device=self.device)
                expected_logp = (q_t * logp).sum()
                div = (q_t * (logp - logp_ref)).sum()
                loss = - expected_logp + self.lambda_kl * div
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}

            if step % eval_interval == 0 or step == 1:
                eval_res = self.eval_fn() if self.eval_fn else {}
                print(f"[MPO] step {step} loss {recorded['loss']:.4f} eval {eval_res}")
                logs.append({"step": step, **recorded, **eval_res})
        return logs


# -----------------------------
# IPO
# -----------------------------
class IPO_RL:
    """
    IPO simplified: learn to increase likelihood of preferred outcomes.
    - Uses logistic loss on log-prob differences (like DPO-ish), but targeted at MLE of preferred events.
    """

    def __init__(self, trainer, beta=1.0, eval_fn=None):
        self.trainer = trainer
        self.beta = beta
        self.eval_fn = eval_fn
        self.device = trainer.device

    def train(self, num_steps=1000, eval_interval=100):
        logs = []
        for step in range(1, num_steps + 1):
            if hasattr(self.trainer, "rollout"):
                trajA = self.trainer.rollout()
                trajB = self.trainer.rollout()
                rA = self.trainer.trajectory_score(trajA)
                rB = self.trainer.trajectory_score(trajB)
                winner = 1.0 if rA >= rB else 0.0
                logpA = self.trainer.logprob_trajectory(trajA)
                logpB = self.trainer.logprob_trajectory(trajB)
                if not isinstance(logpA, torch.Tensor):
                    logpA = torch.tensor(float(logpA), device=self.device)
                if not isinstance(logpB, torch.Tensor):
                    logpB = torch.tensor(float(logpB), device=self.device)
                with torch.no_grad():
                    refA = self.trainer._ref_logprob_trajectory(trajA) if getattr(self.trainer, "_ref_logprob_trajectory", None) else torch.tensor(0.0)
                    refB = self.trainer._ref_logprob_trajectory(trajB) if getattr(self.trainer, "_ref_logprob_trajectory", None) else torch.tensor(0.0)
                if not isinstance(refA, torch.Tensor):
                    refA = torch.tensor(float(refA), device=self.device)
                if not isinstance(refB, torch.Tensor):
                    refB = torch.tensor(float(refB), device=self.device)
                z = self.beta * ((logpA - logpB) - (refA - refB))
                target = torch.tensor([winner], dtype=torch.float32, device=self.device)
                loss = F.binary_cross_entropy_with_logits(z, target)
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}
            else:
                ctx = self.trainer.env.reset()
                acts = self.trainer.sample_candidates(ctx, K=2)
                scores = self.trainer.score_actions(ctx, acts)
                winner = 0 if scores[0] >= scores[1] else 1
                obs = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                acts_t = torch.tensor(np.array(acts), dtype=torch.float32, device=self.device)
                logp = self.trainer.policy.log_prob(obs.repeat(2, 1), acts_t).squeeze(-1)
                with torch.no_grad():
                    logp_ref = self.trainer.ref_policy.log_prob(obs.repeat(2, 1), acts_t).squeeze(-1)
                w_idx = winner
                l_idx = 1 - winner
                z = self.beta * ((logp[w_idx] - logp[l_idx]) - (logp_ref[w_idx] - logp_ref[l_idx]))
                loss = F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                recorded = {"loss": float(loss.item())}

            if step % eval_interval == 0 or step == 1:
                eval_res = self.eval_fn() if self.eval_fn else {}
                print(f"[IPO] step {step} loss {recorded['loss']:.4f} eval {eval_res}")
                logs.append({"step": step, **recorded, **eval_res})
        return logs
