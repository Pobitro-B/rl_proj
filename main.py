# main.py
import argparse
import json
import os
import random
import numpy as np
import torch
from algorithms.pmpo import PMPO
from algorithms.dpo import DPO
from algorithms.mpo import MPO
from algorithms.ipo import IPO


# Import your envs, trainers, and wrappers
from envs.bandit_env import ContextualBandit
from envs.dmcontrol_env import DMControlEnv
from trainers.bandit_trainer import BanditPMPOTrainer
from trainers.dmcontrol_trainer import DMControlPMPOTrainer
from rl_models import PMPO_RL, DPO_RL, MPO_RL, IPO_RL
from models.policy_models import MLPPolicyContinuous
from models.pref_model import ComparatorNet

def make_env(env_type, **kwargs):
    print("main running")
    if env_type == "bandit":
        return ContextualBandit(context_dim=kwargs.get("context_dim", 8),
                                n_actions=kwargs.get("n_actions", 5),
                                reward_fn=kwargs.get("reward_fn", None))
    elif env_type == "dmcontrol":
        return DMControlEnv(domain=kwargs.get("domain","cheetah"),
                            task=kwargs.get("task","run"),
                            from_pixels=kwargs.get("from_pixels", True),
                            height=kwargs.get("height",84),
                            width=kwargs.get("width",84))
    else:
        raise ValueError("Unknown env_type")

def make_policy(env, device="cpu"):
    if hasattr(env, "context_dim"):
        obs_dim = env.context_dim
        act_dim = env.n_actions if hasattr(env, "n_actions") else 2
    else:
        # use simple defaults for DMControl; user should override for their env
        obs_dim = 84*84*3*4 if hasattr(env, "observation_space") else 17
        act_dim = env.action_space.shape[0] if hasattr(env, "action_space") else 2
    policy = MLPPolicyContinuous(obs_dim, act_dim)
    return policy

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Device:", device)

    # create environment
    if args.env == "bandit":
        env = make_env("bandit", context_dim=args.context_dim, n_actions=args.n_actions)
    else:
        env = make_env("dmcontrol", domain=args.domain, task=args.task, from_pixels=args.from_pixels)

    # create models
    policy = make_policy(env)
    ref_policy = make_policy(env)
    pref_model = None
    if args.use_pref_model:
        # comparator input dim should be obs_dim + act_dim (for bandit) or pooled traj dims for DMControl
        obs_dim = env.context_dim if args.env == "bandit" else 512
        act_dim = env.n_actions if args.env == "bandit" else env.action_space.shape[0]
        pref_model = ComparatorNet(obs_dim + act_dim)

    # choose trainer
    if args.env == "bandit":
        trainer = BanditPMPOTrainer(env, policy, ref_policy=ref_policy, pref_model=pref_model,
                                    device=device, lr=args.lr, gamma=args.gamma, lambda_div=args.lambda_div)
    else:
        trainer = DMControlPMPOTrainer(env, policy, ref_policy=ref_policy, pref_model=pref_model,
                                       device=device, gamma=args.gamma, K=args.K, traj_len=args.traj_len,
                                       lambda_div=args.lambda_div, lr=args.lr)

    # evaluation function
    def eval_fn():
        # quick eval: run some rollouts and return mean reward (bandit: average reward over contexts)
        if args.env == "bandit":
            rets = []
            for _ in range(10):
                ctx = env.reset()
                acts = trainer.sample_candidates(ctx, K=8)
                scores = trainer.score_actions(ctx, acts)
                rets.append(max(scores))  # approximate
            return {"eval_mean": float(np.mean(rets))}
        else:
            rets = []
            for _ in range(5):
                traj = trainer.rollout()
                rets.append(sum([s[2] for s in traj]))
            return {"eval_mean": float(np.mean(rets))}

    # choose algorithm wrapper
    alg = args.alg.lower()
    if alg == "pmpo":
        runner = PMPO_RL(trainer, eval_fn=eval_fn)
    elif alg == "dpo":
        runner = DPO_RL(trainer, beta=args.beta, eval_fn=eval_fn)
    elif alg == "mpo":
        runner = MPO_RL(trainer, eta=args.eta, lambda_kl=args.lambda_kl, K=args.K, eval_fn=eval_fn)
    elif alg == "ipo":
        runner = IPO_RL(trainer, beta=args.beta, eval_fn=eval_fn)
    else:
        raise ValueError("Unknown alg")

    logs = runner.train(num_steps=args.steps, eval_interval=args.eval_interval)
    out_dir = args.out_dir or "results"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{alg}_{args.env}_logs.npy"), logs)
    print("Saved logs to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["bandit", "dmcontrol"], default="bandit")
    parser.add_argument("--alg", choices=["pmpo","dpo","mpo","ipo"], default="pmpo")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--context_dim", type=int, default=8)
    parser.add_argument("--n_actions", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--traj_len", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--use_pref_model", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--from_pixels", action="store_true")
    args = parser.parse_args()
    main(args)
