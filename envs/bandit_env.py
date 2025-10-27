# envs/bandit_env.py
import numpy as np

class ContextualBandit:
    """
    Simple contextual bandit environment.
    - context_dim: dimension of context vector
    - n_actions: discrete actions
    - reward_fn: user-supplied function reward(context, action) -> scalar
    """
    def __init__(self, context_dim=8, n_actions=5, reward_fn=None):
        self.context_dim = context_dim
        self.n_actions = n_actions
        self.reward_fn = reward_fn or self.default_reward_fn
        self.rng = np.random.RandomState(0)

    def reset(self):
        # return context
        self.context = self.rng.normal(size=(self.context_dim,))
        return self.context

    def step(self, action):
        reward = self.reward_fn(self.context, action)
        done = True  # single-step episode
        info = {}
        return self.context, reward, done, info

    def default_reward_fn(self, context, action):
        """
        Simple reward combining context and action:
        - Projects both to same dimension
        - Adds mild nonlinearity for variety
        """
        context = np.asarray(context)
        action = np.asarray(action)

        # Project action to context dimension if needed
        if action.shape[0] != context.shape[0]:
            # broadcast or repeat action to match context length
            repeat_factor = int(np.ceil(context.shape[0] / action.shape[0]))
            action = np.tile(action, repeat_factor)[:context.shape[0]]

        # Compute a smooth dot product + a small penalty
        reward = np.tanh(np.dot(context, action)) - 0.01 * np.sum(action**2)
        return float(reward)

