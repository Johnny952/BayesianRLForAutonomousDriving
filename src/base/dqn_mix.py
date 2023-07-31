from base.core import Agent
from rl.policy import Policy
import numpy as np
from collections import deque


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
        info = {
            **action_info_u,
            **action_info,
            **policy_info,
            "max_q_action": np.argmax(q_values),
        }
        return action, info


class MixTestPolicy(Policy):
    def __init__(self, safety_threshold=None, safe_action=None):
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert safe_action is not None

    def reset(self):
        pass

    def select_action(self, q_values, uncertainties):
        if self.safety_threshold is None:
            return np.argmax(q_values), {}
        else:
            sorted_q_indexes = q_values.argsort()[::-1]
            i = 0
            while (
                i < len(uncertainties)
                and uncertainties[sorted_q_indexes[i]] > self.safety_threshold
            ):
                i += 1
            if i == len(
                uncertainties
            ):  # No action is considered safe - use fallback action
                return self.safe_action, {
                    "safe_action": True,
                    "hard_safe": True,
                    "threshold": self.safety_threshold,
                }
            else:
                return sorted_q_indexes[i], {
                    "safe_action": not i == 0,
                    "hard_safe": False,
                    "threshold": self.safety_threshold,
                }


class MixWindowTestPolicy(Policy):
    def __init__(
        self, safety_threshold=None, safe_action=None, history_length=1, start_saving=3
    ) -> None:
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        self.start_saving = start_saving
        self.history = deque([], maxlen=history_length)
        if self.safety_threshold is not None:
            assert safe_action is not None

    def reset(self):
        self.history.clear()

    def add_to_history(self, uncertainty):
        self.history.append(uncertainty)

    def select_action(self, q_values, uncertainties):
        if self.safety_threshold is None:
            return np.argmax(q_values), {}
        else:
            sorted_q_indexes = q_values.argsort()[::-1]
            if len(self.history) < self.start_saving:
                action = np.argmax(q_values)
                self.add_to_history(uncertainties[action])
                return action, {
                    "safe_action": False,
                    "hard_safe": False,
                    "threshold": None,
                }
            threshold = np.mean(self.history) + self.safety_threshold * np.std(
                self.history
            )
            i = 0
            while (
                i < len(uncertainties)
                and uncertainties[sorted_q_indexes[i]] > threshold
            ):
                i += 1
            if i == len(
                uncertainties
            ):  # No action is considered safe - use fallback action
                self.add_to_history(uncertainties[self.safe_action])
                return self.safe_action, {
                    "safe_action": True,
                    "hard_safe": True,
                    "threshold": threshold,
                }
            else:
                self.add_to_history(uncertainties[sorted_q_indexes[i]])
                return sorted_q_indexes[i], {
                    "safe_action": not i == 0,
                    "hard_safe": False,
                    "threshold": threshold,
                }
