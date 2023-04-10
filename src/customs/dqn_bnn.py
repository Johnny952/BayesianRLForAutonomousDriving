import warnings
import torch
import torch.nn as nn
import torchbnn as bnn
from copy import deepcopy
import numpy as np
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import get_object_config
import sys
import wandb
from timeit import default_timer as timer

sys.path.append("..")
from base.core import Agent

def max_q(y_true, y_pred):  # Returns average maximum Q-value of training batch
    return torch.mean(torch.max(y_pred, dim=-1), dim=-1)


def mean_q(y_true, y_pred):  # Returns average Q-value of training batch
    return torch.mean(torch.mean(y_pred, dim=-1), dim=-1)


def clone_model(model):
    return deepcopy(model)


def soft_target_model_updates(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param + (1 - tau) * target_param)
    return target_model


def hard_target_model_updates(target_model, model):
    return soft_target_model_updates(target_model, model, 1)


class AbstractDQNAgent(Agent):
    def __init__(
        self,
        nb_actions,
        memory,
        gamma=0.99,
        batch_size=32,
        nb_steps_warmup=1000,
        train_interval=1,
        memory_interval=1,
        target_model_update=10000,
        delta_range=None,
        delta_clip=np.inf,
        device="cpu",
        **kwargs
    ):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError("`target_model_update` must be >= 0.")
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn(
                "`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we're falling back to `delta_range[1] = {}`".format(
                    delta_range[1]
                )
            )
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.device = device

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch, device):
        batch = self.process_state_batch(state_batch)
        with torch.no_grad():
            q_values = self.model(torch.from_numpy(batch).float().to(device))
            q_values = q_values.cpu().numpy()
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state, device):
        q_values = self.compute_batch_q_values([state], device).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def get_config(self):
        return {
            "nb_actions": self.nb_actions,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "nb_steps_warmup": self.nb_steps_warmup,
            "train_interval": self.train_interval,
            "memory_interval": self.memory_interval,
            "target_model_update": self.target_model_update,
            "delta_clip": self.delta_clip,
            "memory": get_object_config(self.memory),
        }


class DQNBNNAgent(AbstractDQNAgent):
    def __init__(
        self,
        model,
        policy=None,
        test_policy=None,
        enable_double_dqn=True,
        enable_dueling_network=False,
        dueling_type="avg",
        complexity_kld_weight=1,
        sample_forward=10,
        sample_backward=1,
        *args,
        **kwargs
    ):
        super(DQNBNNAgent, self).__init__(*args, **kwargs)

        wandb.init(project="highway-bnn")

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        self.complexity_kld_weight = complexity_kld_weight
        self.sample_forward = sample_forward
        self.sample_backward = sample_backward

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.target_model = None
        self.optimizer = None
        self.loss = None
        self.kl_loss = None
        self.recent_observation = None
        self.recent_action = None

        # State.
        self.reset_states()

        # Counter
        self.backward_nb = 0
        self.forward_nb = 0

    def get_config(self):
        config = super(DQNBNNAgent, self).get_config()
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

        self.loss = nn.HuberLoss(delta=self.delta_clip)
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

        self.compiled = True

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_disct"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_disct"])
        self.target_model = clone_model(self.model)

    def save_weights(self, filepath, overwrite=False):
        tosave = {
            "model_state_disct": self.model.state_dict(),
            "optimizer_state_disct": self.optimizer.state_dict(),
        }
        torch.save(tosave, filepath)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None

    def forward(self, observation):
        tick = timer()
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values_list = []
        for _ in range(self.sample_forward):
            q_values_list.append(self.compute_q_values(state, self.device))
        q_values_list = np.stack(q_values_list)
        q_values = np.mean(q_values_list, axis=0)
        policy_info = {}
        if self.training:
            if hasattr(self.policy, 'custom'):
                action, policy_info = self.policy.select_action(q_values_list)
            else:
                action = self.policy.select_action(q_values=q_values)
        else:
            if hasattr(self.test_policy, 'custom'):
                action, policy_info = self.test_policy.select_action(q_values_list)
            else:
                action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action
        coefficient_of_variation = np.std(q_values_list[:, :], axis=0) / \
                                   np.mean(q_values_list[:, :], axis=0)
        
        # np.sum(np.var(q_values_list, axis=0))
        if self.forward_nb % 1000 == 0:
            tock = timer()
            wandb.log({'Forward time': (tock - tick)})

        self.forward_nb += 1
        policy_info["mean"] = q_values
        policy_info["q_values"] = q_values
        policy_info["coefficient_of_variation"] = coefficient_of_variation
        return action, policy_info

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
            for i in range(self.sample_backward):
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

                if i == 0:
                    q_loss = self.loss(state_action_values, expected_state_action_values)
                else:
                    q_loss += self.loss(state_action_values, expected_state_action_values)
            
            kl_loss = self.kl_loss(self.model)
            loss = q_loss + self.complexity_kld_weight * kl_loss
            if self.backward_nb % 1000 == 0:
                tock = timer()
                wandb.log({'Q Loss': q_loss, 'KL Loss': kl_loss, 'Loss': loss, 'Back time': (tock - tick)})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
