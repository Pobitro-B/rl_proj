# trainers/base_trainer.py
import torch
class BaseTrainer:
    def __init__(self, policy, ref_policy=None, pref_model=None, device='cpu'):
        self.policy = policy.to(device)
        self.ref_policy = (ref_policy.to(device) if ref_policy is not None else None)
        self.pref_model = (pref_model.to(device) if pref_model is not None else None)
        self.device = device

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'pref': self.pref_model.state_dict() if self.pref_model else None
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        if self.pref_model and ck['pref'] is not None:
            self.pref_model.load_state_dict(ck['pref'])
