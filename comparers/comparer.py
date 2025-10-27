# comparers/comparer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(trainer, env, num_steps=1000, eval_interval=100, eval_episodes=10):
    logs = []
    for step in range(num_steps):
        loss = trainer.train_step()
        if step % eval_interval == 0:
            # run evaluation
            returns = []
            for _ in range(eval_episodes):
                traj = trainer.rollout()
                returns.append(sum([s[2] for s in traj]))
            logs.append({'step': step, 'loss': loss, 'mean_return': np.mean(returns)})
    df = pd.DataFrame(logs)
    return df
