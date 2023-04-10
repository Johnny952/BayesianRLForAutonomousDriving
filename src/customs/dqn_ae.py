import torch
import torch.nn as nn
import numpy as np
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import get_object_config
import sys
import wandb
from timeit import default_timer as timer

sys.path.append("..")
from base.core import Agent
from dqn_bnn import max_q, mean_q, clone_model, soft_target_model_updates, hard_target_model_updates, AbstractDQNAgent

class DQNAEAgent(AbstractDQNAgent):
    def __init__(
        self,
        model,
        autoencoder,
        policy=None,
        test_policy=None,
        enable_double_dqn=True,
        enable_dueling_network=False,
        dueling_type="avg",
        update_ae_each=1,
        *args,
        **kwargs
    ):
        super(DQNAEAgent, self).__init__(*args, **kwargs)

        wandb.init(project="highway-ae")

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        self.update_ae_each = update_ae_each

        # Related objects.
        self.model = model
        self.autoencoder = autoencoder
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.target_model = None
        self.optimizer = None
        self.autoencoder_optimizer = None
        self.loss = None
        self.kl_loss = None
        self.recent_observation = None
        self.recent_action = None

        # State.
        self.reset_states()

        # Counter
        self.backward_nb = 0
        self.backward_ae_nb = 0
        self.forward_nb = 0

    def get_config(self):
        config = super(DQNAEAgent, self).get_config()
        config["enable_double_dqn"] = self.enable_double_dqn
        config["dueling_type"] = self.dueling_type
        config["enable_dueling_network"] = self.enable_dueling_network
        config["model"] = get_object_config(self.model)
        config["policy"] = get_object_config(self.policy)
        config["test_policy"] = get_object_config(self.test_policy)
        if self.compiled:
            config["target_model"] = get_object_config(self.target_model)
        return config

    def compile(self, learning_rate, metrics=[]):
        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)

        self.loss = nn.HuberLoss(delta=self.delta_clip)

        self.compiled = True

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_disct"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_disct"])
        self.autoencoder.load_state_dict(checkpoint["autoencoder"])
        self.autoencoder_optimizer.load_state_dict(checkpoint["autoencoder_optimizer"])
        self.target_model = clone_model(self.model)

    def save_weights(self, filepath, overwrite=False):
        tosave = {
            "model_state_disct": self.model.state_dict(),
            "optimizer_state_disct": self.optimizer.state_dict(),
            "autoencoder": self.autoencoder.state_dict(),
            "autoencoder_optimizer": self.autoencoder_optimizer.state_dict(),
        }
        torch.save(tosave, filepath)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None

    def forward(self, observation):
        tick = timer()
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state, self.device)

        uncertainties = []
        action_info = {}
        if self.training:
            if hasattr(self.policy, 'custom'):
                action, action_info = self.policy.select_action(q_values)
            else:
                action = self.policy.select_action(q_values=q_values)

            # Uncertainty for all actions
            obs = torch.from_numpy(observation).unsqueeze(dim=0).float().to(self.device)
            act= torch.Tensor([action]).unsqueeze(dim=0).float().to(self.device)
            with torch.no_grad():
                uncertainty = self.autoencoder.log_prob(obs, act)
            uncertainty = uncertainty.cpu().numpy()
        else:
            # Uncertainty for all actions
            obs = torch.from_numpy(observation).unsqueeze(dim=0).float().to(self.device)
            for i in range(self.nb_actions):
                act = torch.Tensor([i]).unsqueeze(dim=0).float().to(self.device)
                with torch.no_grad():
                    uncertainty = self.autoencoder.log_prob(obs, act)
                uncertainties.append(uncertainty.cpu().numpy())
            if hasattr(self.test_policy, 'custom'):
                action, action_info = self.test_policy.select_action(q_values, uncertainties)
            else:
                action = self.test_policy.select_action(q_values=q_values)
            
            uncertainty = uncertainties[action]

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        if self.forward_nb % 1000 == 0:
            tock = timer()
            wandb.log({'Forward time': (tock - tick), 'Uncertainty': uncertainty})


        self.forward_nb += 1
        action_info["mean"] = q_values
        action_info["q_values"] = q_values
        action_info["coefficient_of_variation"] = np.array(uncertainties)
        return action, action_info

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(
                self.recent_observation,
                self.recent_action,
                reward,
                terminal,
                training=self.training,
            )

        metrics = []
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            tick = timer()
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

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
                terminal1_batch.append(0.0 if e.terminal1 else 1.0)

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

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                state_action_values = self.model(state0_batch)[
                    range(self.batch_size), action_batch
                ].squeeze(dim=-1)
                next_state_values = (
                    terminal1_batch * self.target_model(state1_batch).max(1)[0]
                ).detach()
                expected_state_action_values = (
                    next_state_values * self.gamma
                ) + reward_batch

                # q_values = self.model(torch.from_numpy(state1_batch).float())
                # actions = torch.argmax(q_values, dim=1)
                # target_q_values = self.target_model(torch.from_numpy(state1_batch).float())
                # q_batch = target_q_values[range(self.batch_size), actions]
            else:
                state_action_values = self.target_model(state0_batch).max(dim=-1).detach()
                expected_state_action_values = state_action_values

            loss = self.loss(state_action_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.backward_nb % self.update_ae_each == 0:
                act_batch = torch.from_numpy(np.array(action_batch)).unsqueeze(dim=1).float().to(self.device)
                outputs = self.autoencoder(state0_batch, act_batch)
                auto_loss = self.autoencoder.loss_function(*outputs)

                self.autoencoder_optimizer.zero_grad()
                auto_loss['loss'].backward()
                self.autoencoder_optimizer.step()

                if self.backward_ae_nb % 1000 == 0:
                    tock = timer()
                    wandb.log({'Q Loss': loss, 'Auto Loss': auto_loss['loss'], 'Obs Loss': auto_loss['Obs Loss'], 'Act Loss': auto_loss['Act Loss'], 'Prob Loss': auto_loss['Prob Loss'], 'Back time': (tock - tick)})

                self.backward_ae_nb += 1

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.target_model = hard_target_model_updates(self.target_model, self.model)
        elif self.step % self.target_model_update == 0:
            self.target_model = soft_target_model_updates(
                self.target_model, self.model, self.target_model_update
            )

        self.backward_nb += 1
        return metrics

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
