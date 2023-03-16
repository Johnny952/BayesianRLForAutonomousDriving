"""
Run trained agent on test episodes or special cases.

This script should be called from the folder src, where the script is located.

The options for what to run are set below, after the import statements. The options are:
- Which agent to run, defined by:
   - filepath = '../logs/train_agent_DATE_TIME_NAME/', which is the folder name of log file for a specific run
   - agent name = 'STEP', which chooses the agent that was saved after STEP training steps.
                          In case of ensemble, only use STEP and do not include _N, where N is the ensemble index.
- Which case to run:
   - case = 'CASE_NAME', options 'rerun_test_scenarios', 'fast_overtaking', 'standstill'
- Ensemble test policy:
   - use_ensemble_test_policy = True/False, which specifies it the ensemble safety policy should be used or not
   - safety_threshold = e.g. 0.02, which defines the maximum allowed coefficient of variation for an action
- Save run as video:
   - save_video = True/False, if True, a set of images are stored in '../Videos/', which can be used to create a video

The different cases are:
- rerun_test_scenarios: This option will run the trained agent on the same test episodes that are used to evaluate
                        the performance of the agent during the training process.
- fast_overtaking: This option demonstrates how the safety criterion of the ensemble RPF agent can be used,
                   and is the same as was included in the paper.
- standstill: This option demonstrates how the safety criterion of the ensemble RPF agent can be used,
              and is the same as was included in the paper.
"""

import sys
import csv
import numpy as np
from rl.policy import GreedyQPolicy
from rl.memory import Memory

sys.path.append("..")
from dqn_bnn import DQNBNNAgent
from dqn_ae import DQNAEAgent
from network_architecture import NetworkMLPBNN, NetworkCNNBNN, NetworkAE, NetworkMLP, NetworkCNN
from base.driving_env import Highway
import traci
from matplotlib import rcParams
from safe_greedy_policy import SafeGreedyPolicy

rcParams['pdf.fonttype'] = 42   # To avoid Type 3 fonts in figures
rcParams['ps.fonttype'] = 42

""" Options: """
filepath = "../logs/bnn_train_agent_20230210_195642/"
agent_name = "4950061"
case = "standstill"  # 'rerun_test_scenarios', 'fast_overtaking', 'standstill'
safety_threshold = 0.0045  # Only used if ensemble test policy is chosen 0.0045
use_safe_action = True

thresh_range = [0.01, 0.05]
thresh_steps = 50

# test scenarios
nb_vehicles = 30
speed_range = [10, 20]

# Standstill
stop_position_range = [200, 400]
stop_speed_range = [0, 5]
""" End options """

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + 'src/')
import parameters_stored as p
import parameters_simulation_stored as ps

ps.sim_params["remove_sumo_warnings"] = False
nb_actions = len(ps.sim_params["action_interp"])

if case == "rerun_test_scenarios":
    ps.sim_params["nb_vehicles"] = nb_vehicles
    ps.road_params["speed_range"] = np.array(speed_range)
elif case == "fast_overtaking":
    ps.sim_params["nb_vehicles"] = 5
elif case == "standstill":
    ps.sim_params["nb_vehicles"] = 7
else:
    raise Exception("Case not defined.")

env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False, return_more_info=True)

nb_observations = env.nb_observations
if p.agent_par["model"] == 'bnn':
    if p.agent_par["cnn"]:
        model = NetworkCNNBNN(
            env.nb_ego_states,
            env.nb_states_per_vehicle,
            ps.sim_params["sensor_nb_vehicles"],
            nb_actions,
            nb_conv_layers=p.agent_par["nb_conv_layers"],
            nb_conv_filters=p.agent_par["nb_conv_filters"],
            nb_hidden_fc_layers=p.agent_par["nb_hidden_fc_layers"],
            nb_hidden_neurons=p.agent_par["nb_hidden_neurons"],
            duel=p.agent_par["duel_q"],
            prior=False,
            activation="relu",
            window_length=p.agent_par["window_length"],
            duel_type="avg",
            prior_mu=p.agent_par["prior_mu"],
            prior_sigma=p.agent_par["prior_sigma"],
        ).to(p.agent_par["device"])
    else:
        model = NetworkMLPBNN(
            nb_observations,
            nb_actions,
            nb_hidden_layers=p.agent_par["nb_hidden_fc_layers"],
            nb_hidden_neurons=p.agent_par["nb_hidden_neurons"],
            duel=p.agent_par["duel_q"],
            prior=False,
            activation="relu",
            duel_type="avg",
            window_length=p.agent_par["window_length"],
            prior_mu=p.agent_par["prior_mu"],
            prior_sigma=p.agent_par["prior_sigma"],
        ).to(p.agent_par["device"])

    memory = Memory(
        window_length=p.agent_par["window_length"]
    )  # Not used, simply needed to create the agent

    policy = GreedyQPolicy()
    safety_threshold_ = safety_threshold if use_safe_action else None
    safe_action = 3 if use_safe_action else None
    test_policy = SafeGreedyPolicy(policy_type='mean', safety_threshold=safety_threshold_, safe_action=safe_action)
    dqn = DQNBNNAgent(
        model=model,
        policy=policy,
        test_policy=test_policy,
        enable_double_dqn=p.agent_par["double_q"],
        enable_dueling_network=False,
        nb_actions=nb_actions,
        memory=memory,
        gamma=p.agent_par["gamma"],
        batch_size=p.agent_par["batch_size"],
        nb_steps_warmup=p.agent_par["learning_starts"],
        train_interval=p.agent_par["train_freq"],
        target_model_update=p.agent_par["target_network_update_freq"],
        delta_clip=p.agent_par["delta_clip"],
        complexity_kld_weight=p.agent_par["complexity_kld_weight"],
        sample_forward=p.agent_par["sample_forward"],
        sample_backward=p.agent_par["sample_backward"],
    )
elif p.agent_par["model"] == 'ae':
    ae = NetworkAE(
        p.agent_par["window_length"],
        nb_observations,
        nb_actions,
        obs_encoder_arc=p.agent_par["obs_encoder_arc"],
        act_encoder_arc=p.agent_par["act_encoder_arc"],
        shared_encoder_arc=p.agent_par["shared_encoder_arc"],
        obs_decoder_arc=p.agent_par["obs_decoder_arc"],
        act_decoder_arc=p.agent_par["act_decoder_arc"],
        shared_decoder_arc=p.agent_par["shared_decoder_arc"],
        covar_decoder_arc=p.agent_par["covar_decoder_arc"],
        latent_dim=p.agent_par["latent_dim"],
        act_loss_weight=p.agent_par["act_loss_weight"],
        obs_loss_weight=p.agent_par["obs_loss_weight"],
        prob_loss_weight=p.agent_par["prob_loss_weight"],
    ).to(p.agent_par['device'])
    if p.agent_par["cnn"]:
        model = NetworkCNN(
            env.nb_ego_states,
            env.nb_states_per_vehicle,
            ps.sim_params["sensor_nb_vehicles"],
            nb_actions,
            nb_conv_layers=p.agent_par["nb_conv_layers"],
            nb_conv_filters=p.agent_par["nb_conv_filters"],
            nb_hidden_fc_layers=p.agent_par["nb_hidden_fc_layers"],
            nb_hidden_neurons=p.agent_par["nb_hidden_neurons"],
            duel=p.agent_par["duel_q"],
            prior=False,
            activation="relu",
            window_length=p.agent_par["window_length"],
            duel_type="avg",
        ).to(p.agent_par["device"])
    else:
        model = NetworkMLP(
            nb_observations,
            nb_actions,
            nb_hidden_layers=p.agent_par["nb_hidden_fc_layers"],
            nb_hidden_neurons=p.agent_par["nb_hidden_neurons"],
            duel=p.agent_par["duel_q"],
            prior=False,
            activation="relu",
            duel_type="avg",
            window_length=p.agent_par["window_length"],
        ).to(p.agent_par["device"])

    memory = Memory(
        window_length=p.agent_par["window_length"]
    )  # Not used, simply needed to create the agent

    policy = GreedyQPolicy()
    test_policy = GreedyQPolicy()
    dqn = DQNAEAgent(
        model,
        ae,
        policy,
        test_policy,
        enable_double_dqn=p.agent_par["double_q"],
        enable_dueling_network=False,
        nb_actions=nb_actions,
        memory=memory,
        gamma=p.agent_par["gamma"],
        batch_size=p.agent_par["batch_size"],
        nb_steps_warmup=p.agent_par["learning_starts"],
        train_interval=p.agent_par["train_freq"],
        target_model_update=p.agent_par["target_network_update_freq"],
        delta_clip=p.agent_par["delta_clip"],
        device=p.agent_par["device"],
        update_ae_each=p.agent_par["update_ae_each"]
    )
else:
    raise Exception("Model not implemented.")
dqn.compile(p.agent_par["learning_rate"])

dqn.load_weights(filepath + agent_name)
dqn.training = False

with open(filepath + case + '.csv', 'w+') as file:
    pass

if case == 'rerun_test_scenarios':
    env.reset()
    for thresh in np.linspace(thresh_range[0], thresh_range[1], num=thresh_steps):
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        collissions = 0
        if use_safe_action:
            dqn.test_policy = SafeGreedyPolicy(policy_type='mean', safety_threshold=thresh, safe_action=safe_action)
        for i in range(0, 100):
            np.random.seed(i)
            obs = env.reset()
            done = False
            episode_reward = 0
            step = 0
            nb_safe_actions = 0
            max_coef_of_var = 0
            while done is False:
                action, action_info = dqn.forward(obs)
                obs, rewards, done, info, more_info = env.step(action, action_info)
                episode_reward += rewards
                step += 1
                if more_info["ego_collision"] == True:
                    collissions += 1
                if 'safe_action' in action_info:
                    if action_info['safe_action']:
                        nb_safe_actions += 1
                    if action_info['coefficient_of_variation'][action] > max_coef_of_var:
                        max_coef_of_var = action_info['coefficient_of_variation'][action]

            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            print(f"Episode: {str(i)}\tSteps: {str(step)}\tReward: {str(episode_reward)}\tNumber of safety actions: {str(nb_safe_actions)}\tMax coef of var: {str(max_coef_of_var)}")
        print(f"\nThreshold: {str(thresh)}\tMean Reward: {str(np.mean(episode_rewards))}\tCollision Rate: {str(collissions)}%\n\n")
        with open(filepath + case + '.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow([thresh, np.sum(episode_rewards) / np.sum(episode_steps), collissions / 100])

elif case == 'fast_overtaking':
    env.reset()

    for thresh in np.linspace(thresh_range[0], thresh_range[1], num=thresh_steps):
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        collissions = 0
        if use_safe_action:
            dqn.test_policy = SafeGreedyPolicy(policy_type='mean', safety_threshold=thresh, safe_action=safe_action)
        for i in range(0, 100):
            # Make sure that the vehicles are not affected by previous state
            np.random.seed(57)
            env.reset()
            s0 = 1000.
            traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
            traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
            traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
            traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
            traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
            traci.simulationStep()
            env.speeds[0, 0] = 15
            for veh in env.vehicles:
                traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setSpeed('veh0', 15)
            traci.vehicle.setSpeed('veh1', 15)
            traci.vehicle.setSpeed('veh2', 55)
            traci.vehicle.setSpeed('veh3', 15)
            traci.vehicle.setSpeed('veh4', 15)
            traci.vehicle.setMaxSpeed('veh0', 25)
            traci.vehicle.setMaxSpeed('veh1', 15)
            traci.vehicle.setMaxSpeed('veh2', 55)
            traci.vehicle.setMaxSpeed('veh3', 15)
            traci.vehicle.setMaxSpeed('veh4', 15)
            traci.simulationStep()
            env.step(0)

            # Overtaking case
            traci.vehicle.moveTo('veh0', 'highway_0', s0)
            traci.vehicle.moveTo('veh1', 'highway_0', s0 + 50)
            traci.vehicle.moveTo('veh2', 'highway_1', s0 - 150)
            traci.vehicle.moveTo('veh3', 'highway_2', s0 - 50)
            traci.vehicle.moveTo('veh4', 'highway_2', s0 - 0)
            traci.vehicle.setSpeed('veh0', 15)
            traci.vehicle.setSpeed('veh1', 15)
            traci.vehicle.setSpeed('veh2', 55)
            traci.vehicle.setSpeed('veh3', 15)
            traci.vehicle.setSpeed('veh4', 15)
            observation, reward, done, info, more_info = env.step(0)
            for veh in env.vehicles[1:]:
                traci.vehicle.setSpeed(veh, -1)

            # Run fast overtaking case
            action_log = []
            q_log = []
            cv_log = []
            v_log = []
            nb_safe_actions = 0
            episode_reward = 0
            step = 0
            max_coef_of_var = 0
            for _ in range(8):
                action, action_info = dqn.forward(observation)
                observation, reward, done, info, more_info = env.step(action, action_info)
                episode_reward += reward
                step += 1
                if more_info["ego_collision"] == True:
                    collissions += 1
                if 'safe_action' in action_info:
                    if action_info['safe_action']:
                        nb_safe_actions += 1
                action_log.append(action)
                q_log.append(action_info['mean'] if p.agent_par['ensemble'] else action_info['q_values'])
                cv_log.append(action_info['coefficient_of_variation'] if p.agent_par['ensemble'] else action_info['q_values']*0)
                v_log.append(env.speeds[0, 0])
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            print(f"Episode: {str(i)}\tSteps: {str(step)}\tReward: {str(episode_reward)}\tNumber of safety actions: {str(nb_safe_actions)}\tMax coef of var: {str(max_coef_of_var)}")
        print(f"\nThreshold: {str(thresh)}\tMean Reward: {str(np.mean(episode_rewards))}\tCollision Rate: {str(collissions)}%\n\n")
        with open(filepath + case + '.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow([thresh, np.sum(episode_rewards) / np.sum(episode_steps), collissions / 100])


elif case == 'standstill':
    env.reset()

    for thresh in np.linspace(thresh_range[0], thresh_range[1], num=thresh_steps):
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        collissions = 0
        if use_safe_action:
            dqn.test_policy = SafeGreedyPolicy(policy_type='mean', safety_threshold=thresh, safe_action=safe_action)
        for i in range(0, 100):
            # Make sure that the vehicles are not affected by previous state
            np.random.seed(57)
            env.reset()
            s0 = 1000.
            stop_speed = np.random.uniform(stop_speed_range[0], stop_speed_range[1])
            traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
            traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
            traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
            traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
            traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
            traci.vehicle.moveTo('veh5', 'highway_2', s0 - 600)
            traci.vehicle.moveTo('veh6', 'highway_2', s0 - 700)
            traci.simulationStep()
            env.speeds[0, 0] = 25
            for veh in env.vehicles:
                traci.vehicle.setSpeedMode(veh, 0)
                traci.vehicle.setLaneChangeMode(veh, 0)  # Turn off all lane changes
            traci.vehicle.setSpeed('veh0', 25)
            traci.vehicle.setSpeed('veh1', stop_speed)
            traci.vehicle.setSpeed('veh2', 15)
            traci.vehicle.setSpeed('veh3', 15)
            traci.vehicle.setSpeed('veh4', 15)
            traci.vehicle.setSpeed('veh5', 15)
            traci.vehicle.setSpeed('veh6', 15)
            traci.simulationStep()
            traci.vehicle.setMaxSpeed('veh0', 25)
            traci.vehicle.setMaxSpeed('veh1', stop_speed)
            traci.vehicle.setMaxSpeed('veh2', 15)
            traci.vehicle.setMaxSpeed('veh3', 15)
            traci.vehicle.setMaxSpeed('veh4', 15)
            traci.vehicle.setMaxSpeed('veh5', 15)
            traci.vehicle.setMaxSpeed('veh6', 15)
            traci.simulationStep()
            env.step(0)

            # Standstill case
            traci.vehicle.moveTo('veh0', 'highway_0', s0)
            traci.vehicle.moveTo('veh1', 'highway_0', s0 + np.random.uniform(stop_position_range[0], stop_position_range[1]))
            traci.vehicle.moveTo('veh2', 'highway_1', s0 + 36)
            traci.vehicle.moveTo('veh3', 'highway_1', s0 + 54)
            traci.vehicle.moveTo('veh4', 'highway_1', s0 + 72)
            traci.vehicle.moveTo('veh5', 'highway_1', s0 + 90)
            traci.vehicle.moveTo('veh6', 'highway_1', s0 + 108)
            traci.vehicle.setSpeed('veh0', 25)
            traci.vehicle.setSpeed('veh1', stop_speed)
            traci.vehicle.setSpeed('veh2', 15)
            traci.vehicle.setSpeed('veh3', 15)
            traci.vehicle.setSpeed('veh4', 15)
            traci.vehicle.setSpeed('veh5', 15)
            traci.vehicle.setSpeed('veh6', 15)
            observation, reward, done, info, more_info = env.step(0)
            for veh in env.vehicles[1:]:
                traci.vehicle.setSpeed(veh, -1)

            # Run standstill case
            action_log = []
            q_log = []
            cv_log = []
            v_log = []
            nb_safe_actions = 0

            episode_reward = 0
            step = 0
            max_coef_of_var = 0
            for _ in range(30):
                action, action_info = dqn.forward(observation)
                observation, reward, done, info, more_info = env.step(action)
                episode_reward += reward
                step += 1
                if more_info["ego_collision"] == True:
                    collissions += 1
                if 'safe_action' in action_info:
                    if action_info['safe_action']:
                        nb_safe_actions += 1
                action_log.append(action)
                q_log.append(action_info['mean'] if p.agent_par['ensemble'] else action_info['q_values'])
                cv_log.append(action_info['coefficient_of_variation'] if p.agent_par['ensemble'] else action_info['q_values']*0)
                v_log.append(env.speeds[0, 0])
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            print(f"Episode: {str(i)}\tSteps: {str(step)}\tReward: {str(episode_reward)}\tNumber of safety actions: {str(nb_safe_actions)}\tMax coef of var: {str(max_coef_of_var)}")
        print(f"\nThreshold: {str(thresh)}\tMean Reward: {str(np.mean(episode_rewards))}\tCollision Rate: {str(collissions)}%\n\n")
        with open(filepath + case + '.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow([thresh, np.sum(episode_rewards) / np.sum(episode_steps), collissions / 100])

else:
    raise Exception('Case not defined.')
