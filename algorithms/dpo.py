import torch
import torch.nn.functional as F

class DPO:
    """
    Direct Preference Optimization
    L = -E[ log σ(β(Δlogπ - Δlogπ_ref)) ]
    """

    def __init__(self, policy, ref_policy, optimizer, beta=1.0):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.beta = beta

    def train_step(self, obs_pref, obs_nonpref, acts_pref, acts_nonpref):
        logp_pref = self.policy.log_prob(obs_pref, acts_pref)
        logp_nonpref = self.policy.log_prob(obs_nonpref, acts_nonpref)
        with torch.no_grad():
            logp_ref_pref = self.ref_policy.log_prob(obs_pref, acts_pref)
            logp_ref_nonpref = self.ref_policy.log_prob(obs_nonpref, acts_nonpref)

        z = self.beta * ((logp_pref - logp_nonpref) - (logp_ref_pref - logp_ref_nonpref))
        loss = -F.logsigmoid(z).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
