# envs/wrappers.py
import numpy as np
from collections import deque

class RGBStackWrapper:
    """
    Stack the last K RGB frames along channel dimension.
    Expects env to return observations as HxWxC (0-255) np.uint8.
    """
    def __init__(self, env, K=4):
        self.env = env
        self.K = K
        obs = env.reset()
        h, w, c = obs.shape
        self.shape = (h, w, c * K)
        self.frames = deque(maxlen=K)
        for _ in range(K):
            self.frames.append(np.zeros((h, w, c), dtype=np.uint8))

    def reset(self):
        obs = self.env.reset()
        self.frames.clear()
        for _ in range(self.K - 1):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # concat along channel axis
        return np.concatenate(list(self.frames), axis=2)
