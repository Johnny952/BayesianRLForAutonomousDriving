from base.core import Agent
from rl.policy import Policy
from base.dqn_ensemble import DQNAgentEnsembleParallel
import numpy as np
import torch
import wandb


class RPFDAEAgent(DQNAgentEnsembleParallel):
    def __init__(self, update_ae_each, nb_models, learning_rate, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_conv_layers, nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, network_seed, prior_scale_factor, window_length, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg', *args, **kwargs):
        super().__init__(nb_models, learning_rate, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_conv_layers, nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, network_seed, prior_scale_factor, window_length, policy, test_policy, enable_double_dqn, enable_dueling_network, dueling_type, *args, **kwargs)
        wandb.init(project="highway-rpf-dae")
        self.backward_dae_nb = 0
        self.nb_backwards = 0
        self.update_ae_each = update_ae_each

    def set_uncertainty_model(self, model, optimizer, device):
        self.u_model = model
        self.u_optimizer = optimizer
        self.device = device

    def save_weights(self, filepath, overwrite=False):
        super().save_weights(filepath, overwrite)

        tosave = {
            "model_state_disct": self.u_model.state_dict(),
            "optimizer_state_disct": self.u_optimizer.state_dict(),
        }
        torch.save(tosave, filepath + '_dae')
    
    def load_weights(self, filepath):
        super().load_weights(filepath)

        checkpoint = torch.load(filepath + '_dae', map_location=torch.device(self.device))
        self.u_model.load_state_dict(checkpoint["model_state_disct"])
        self.u_optimizer.load_state_dict(checkpoint["optimizer_state_disct"])

    def forward(self, observation):
        action, info = super().forward(observation)

        uncertainties = []
        obs = torch.from_numpy(observation).unsqueeze(dim=0).float().to(self.device)
        with torch.no_grad():
            uncertainties = []
            
            for i in range(self.nb_actions):
                act = torch.Tensor([i]).unsqueeze(dim=0).float().to(self.device)
                [obs_mu_i, act_mu_i, covar_i, (obs_i, act_i)] = self.u_model(obs, act)
                nll = self.u_model.nll_loss(obs_mu_i, obs_i, act_mu_i, act_i, covar_i)
                nll_obs = self.u_model.obs_nll_loss(obs_mu_i, obs_i, covar_i)
                uncertainties.append(nll_obs + 1 * (nll - nll_obs))

        unc = np.array([u.data.cpu().numpy() for u in uncertainties])
        if not self.training and hasattr(self.test_policy, "custom"):
            action, action_info = self.test_policy.select_action(
                info['q_values_all_nets'], unc
            )
        
        info["coefficient_of_variation"] = unc

        return action, {**info, **action_info}

    def update_dae(self, experiences):
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        state0_batch = torch.from_numpy(state0_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        # action_batch = torch.from_numpy(action_batch).long().to(self.device)
        terminal1_batch = torch.from_numpy(terminal1_batch).float().to(self.device)
        state1_batch = torch.from_numpy(state1_batch).float().to(self.device)
    
        act_batch = (
            torch.from_numpy(np.array(action_batch))
            .unsqueeze(dim=1)
            .float()
            .to(self.device)
        )

        output = self.u_model(state0_batch, act_batch)
        ae_loss = self.u_model.loss_function(*output)
        self.u_optimizer.zero_grad()
        ae_loss["loss"].backward()
        self.u_optimizer.step()

        if self.backward_dae_nb % 1000 == 0:
            wandb.log(
                {
                    "Auto Loss": ae_loss["loss"],
                    "Obs Loss": ae_loss["Obs Loss"],
                    "Act Loss": ae_loss["Act Loss"],
                    "Prob Loss": ae_loss["Prob Loss"],
                }
            )

        self.backward_dae_nb += 1
    
    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if self.training:
            if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
                for net in range(self.nb_models):
                    experiences = self.memory.sample(net, self.batch_size)
                    assert len(experiences) == self.batch_size
                    self.input_queues[net].put(['train', experiences])
                    if self.nb_backwards % self.update_ae_each == 0:
                        self.update_dae(experiences)

                for net in range(self.nb_models):   # Wait for all workers to finish
                    output = self.output_queues[net].get()
                    if net == self.nb_models - 1:   # Store the metrics of the last agent
                        metrics = output[1]
                    assert(output[0] == 'training_done_' + str(net))

                    metrics += [self.active_model]
            
            self.nb_backwards += 1

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics   # This is only the metrics of the last agent.


    
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

class MixBaseDQNPolicy(Policy):
    def __init__(self, q_model):
        self.q_model = q_model
    
    def select_action(self, q_values):
        return self.q_model.select_action(q_values)