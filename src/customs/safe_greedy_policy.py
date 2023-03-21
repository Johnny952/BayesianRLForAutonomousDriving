from rl.policy import GreedyQPolicy
import numpy as np


class SafeGreedyPolicy(GreedyQPolicy):
    """
    Policy used by the ensemble method during testing episodes.

    During testing episodes, the policy chooses the action with either the highest mean Q-value or the actions that has
    the highest Q-value in most of the ensemble members.
    If safety_threshold is set, only actions with a coefficient of variation below the set value are considered.
    If no action is considered safe, the fallback action safe_action is used.

    Args:
        policy_type (str): 'mean' or 'voting'
        safety_threshold (float): Maximum coefficient of variation that is considered safe.
        safe_action (int): Fallback action if all actions are considered unsafe.
    """

    def __init__(self, policy_type='mean', safety_threshold=None, safe_action=None):
        super(SafeGreedyPolicy, self).__init__()
        self.custom = True
        self.policy_type = policy_type
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert(safe_action is not None)

    def select_action(self, q_values):
        if self.policy_type == 'mean':
            mean_q_values = np.mean(q_values, axis=0)
            if self.safety_threshold is None:
                return np.argmax(mean_q_values), {}
            else:
                std_q_values = np.std(q_values, axis=0)
                coef_of_var = std_q_values / np.abs(mean_q_values)
                sorted_q_indexes = mean_q_values.argsort()[::-1]
                i = 0
                while i < len(coef_of_var) and coef_of_var[sorted_q_indexes[i]] > self.safety_threshold:
                    i += 1
                if i == len(coef_of_var):  # No action is considered safe - use fallback action
                    return self.safe_action, {'safe_action': True}
                else:
                    return sorted_q_indexes[i], {'safe_action': not i == 0}
        elif self.policy_type == 'voting':
            action_votes = np.argmax(q_values, axis=1)
            actions, counts = np.unique(action_votes, return_counts=True)
            max_actions = np.flatnonzero(counts == max(counts))
            action = actions[np.random.choice(max_actions)]
            if self.safety_threshold is None:
                return action, {}
            else:
                raise Exception('Voting policy for safe actions is not yet implemented.')
        else:
            raise Exception('Unvalid policy type defined.')

    def get_config(self):
        config = super(SafeGreedyPolicy, self).get_config()
        config['type'] = self.policy_type
        return config


class SimpleSafeGreedyPolicy(GreedyQPolicy):
    def __init__(self, safety_threshold=None, safe_action=None):
        super(SimpleSafeGreedyPolicy, self).__init__()
        self.custom = True
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert(safe_action is not None)

    def select_action(self, q_values, uncertainties):
        act = np.argmax(q_values)
        if self.safety_threshold is None:
            return act, {}
        else:
            if uncertainties[act] > self.safety_threshold:  # No action is considered safe - use fallback action
                return self.safe_action, {'safe_action': True}
            else:
                return act, {'safe_action': False}
