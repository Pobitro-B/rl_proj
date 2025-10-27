# envs/dmcontrol_env.py
try:
    import dmc2gym
except Exception as e:
    dmc2gym = None

import numpy as np

class DMControlEnv:
    def __init__(self, domain='cheetah', task='run', from_pixels=True, height=84, width=84, frame_skip=2):
        assert dmc2gym is not None, "Install dmc2gym or use dm_control"
        self.env = dmc2gym.make(domain_name=domain, task_name=task, seed=0, visualize_reward=False, from_pixels=from_pixels,
                                height=height, width=width, frame_skip=frame_skip)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
