from base.core import Agent
from rl.policy import Policy
import numpy as np

class MixDQNAgent(Agent):
    def __init__(self, q_model, u_model, policy, **kwargs):
        super(MixDQNAgent, self).__init__(**kwargs)

        self.q_model = q_model
        self.u_model = u_model
        self.policy = policy

    def forward(self, observation):
        _, action_info = self.q_model.forward(observation)
        q_values = action_info["q_values"]
        _, action_info_u = self.u_model.forward(observation)
        if "coefficient_of_variation" in action_info_u:
            uncertainties = action_info_u["coefficient_of_variation"]
        else:
            uncertainties = np.ones_like(q_values)
        action, policy_info = self.policy.select_action(q_values, uncertainties)
        info = {**action_info_u, **action_info, **policy_info}
        return action, info
    



class MixTestPolicy(Policy):
    def __init__(self, safety_threshold=None, safe_action=None):
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert(safe_action is not None)

    def select_action(self, q_values, uncertainties):
        if self.safety_threshold is None:
            return np.argmax(q_values), {}
        else:
            sorted_q_indexes = q_values.argsort()[::-1]
            i = 0
            while i < len(uncertainties) and np.abs(uncertainties[sorted_q_indexes[i]]) > self.safety_threshold:
                i += 1
            if i == len(uncertainties):  # No action is considered safe - use fallback action
                return self.safe_action, {'safe_action': True, 'hard_safe': True}
            else:
                return sorted_q_indexes[i], {'safe_action': not i == 0, 'hard_safe': False}
