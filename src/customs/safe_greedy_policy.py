from rl.policy import GreedyQPolicy
import numpy as np


class SafeGreedyPolicy(GreedyQPolicy):

    def __init__(self, safety_threshold=None, safe_action=None):
        super(SafeGreedyPolicy, self).__init__()
        self.custom = True
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert safe_action is not None

    def select_action(self, q_values, q_values_std):
        if self.safety_threshold is None:
            return np.argmax(q_values), {}
        else:
            coef_of_var = np.abs(q_values_std - q_values)
            sorted_q_indexes = q_values.argsort()[::-1]
            i = 0
            while (
                i < len(coef_of_var)
                and coef_of_var[sorted_q_indexes[i]] > self.safety_threshold
            ):
                i += 1
            if i == len(
                coef_of_var
            ):  # No action is considered safe - use fallback action
                return self.safe_action, {"safe_action": True, "hard_safe": True}
            else:
                return sorted_q_indexes[i], {
                    "safe_action": not i == 0,
                    "hard_safe": False,
                }

    def get_config(self):
        config = super(SafeGreedyPolicy, self).get_config()
        config["type"] = self.policy_type
        return config


class SimpleSafeGreedyPolicy(GreedyQPolicy):
    def __init__(self, safety_threshold=None, safe_action=None):
        super(SimpleSafeGreedyPolicy, self).__init__()
        self.custom = True
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert safe_action is not None

    def select_action(self, q_values, uncertainties):
        if self.safety_threshold is None:
            return np.argmax(q_values), {}
        else:
            sorted_q_indexes = q_values.argsort()[::-1]
            i = 0
            while i < len(uncertainties) and uncertainties[sorted_q_indexes[i]] > self.safety_threshold:
                i += 1
            if i == len(
                uncertainties
            ):  # No action is considered safe - use fallback action
                return self.safe_action, {"safe_action": True, "hard_safe": True}
            else:
                return sorted_q_indexes[i], {
                    "safe_action": not i == 0,
                    "hard_safe": False,
                }
