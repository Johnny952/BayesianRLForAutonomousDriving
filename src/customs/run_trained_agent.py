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

import os
import sys
import numpy as np
from rl.policy import GreedyQPolicy
from rl.memory import Memory

sys.path.append("..")
from dqn_bnn import DQNBNNAgent
from dqn_ae import DQNAEAgent
from network_architecture import NetworkMLPBNN, NetworkCNNBNN, NetworkAE, NetworkMLP, NetworkCNN
from base.driving_env import Highway
import traci
import matplotlib.pyplot as plt
from matplotlib import rcParams
from safe_greedy_policy import SafeGreedyPolicy, SimpleSafeGreedyPolicy

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

""" Options: """
filepath = "../logs/train_agent_20230329_191111_bnn_6M_v3/"
name = 'bnn'
agent_name = "5950017"
case = "standstill"  # 'rerun_test_scenarios', 'fast_overtaking', 'standstill'
safety_threshold = 0.3  # Only used if ensemble test policy is chosen BNN: 0.0045, AE: 0.7
save_video = True
use_safe_action = True
""" End options """

label = 'U' if use_safe_action else 'NU'

# These import statements need to come after the choice of Falsewhich agent that should be used.
sys.path.insert(0, filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps


ps.sim_params["remove_sumo_warnings"] = False
nb_actions = len(ps.sim_params["action_interp"])

if case == "rerun_test_scenarios":
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
elif case == "fast_overtaking":
    ps.sim_params["nb_vehicles"] = 5
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
elif case == "standstill":
    ps.sim_params["nb_vehicles"] = 7
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
else:
    raise Exception("Case not defined.")

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
    if use_safe_action:
        safety_threshold_ = safety_threshold
        safe_action = 3
        test_policy = SimpleSafeGreedyPolicy(safety_threshold_, safe_action)
    else:
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

if save_video and not os.path.isdir("../videos"):
    os.mkdir("../videos")

if case == "rerun_test_scenarios":
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema("View #0", "scheme_for_videos")
        except Exception as e:
            print("SUMO scheme not defined.")
        if not os.path.isdir("../videos"):
            os.mkdir("../videos")
    episode_rewards = []
    episode_steps = []
    nb_safe_actions_per_episode = []
    for i in range(0, 100):
        np.random.seed(i)
        obs = env.reset()
        if save_video:
            video_folder = "../videos/rerun_" + str(i) + "___" + filepath[8:-1]
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom("View #0", 10500)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        done = False
        episode_reward = 0
        step = 0
        nb_safe_actions = 0
        max_coef_of_var = 0
        while done is False:
            action, action_info = dqn.forward(obs)
            obs, rewards, done, info = env.step(action, action_info)
            episode_reward += rewards
            step += 1
            if "safe_action" in action_info:
                if action_info["safe_action"]:
                    nb_safe_actions += 1
                if action_info["coefficient_of_variation"][action] > max_coef_of_var:
                    max_coef_of_var = action_info["coefficient_of_variation"][action]
            if save_video:
                traci.gui.screenshot(
                    "View #0", video_folder + "/images/" + str(step) + ".png"
                )

        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        nb_safe_actions_per_episode.append(nb_safe_actions)
        print("Episode: " + str(i))
        print("Episode steps: " + str(step))
        print("Episode reward: " + str(episode_reward))
        print("Number of safety actions: " + str(nb_safe_actions))
        print("Max coef of var: " + str(max_coef_of_var))

    print(episode_rewards)
    print(episode_steps)
    print(nb_safe_actions_per_episode)

elif case == "fast_overtaking":
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema("View #0", "scheme_for_videos")
        except Exception as e:
            print("SUMO scheme not defined.")
    for _ in range(5):  # Display the case 5 times
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)
        env.reset()
        if env.use_gui:
            traci.vehicle.setColor("veh2", (178, 102, 255))
        s0 = 1000.0
        traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
        traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
        traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
        traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
        traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
        traci.simulationStep()
        env.speeds[0, 0] = 15
        for veh in env.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
        traci.vehicle.setSpeed("veh0", 15)
        traci.vehicle.setSpeed("veh1", 15)
        traci.vehicle.setSpeed("veh2", 55)
        traci.vehicle.setSpeed("veh3", 15)
        traci.vehicle.setSpeed("veh4", 15)
        traci.vehicle.setMaxSpeed("veh0", 25)
        traci.vehicle.setMaxSpeed("veh1", 15)
        traci.vehicle.setMaxSpeed("veh2", 55)
        traci.vehicle.setMaxSpeed("veh3", 15)
        traci.vehicle.setMaxSpeed("veh4", 15)
        traci.simulationStep()
        env.step(0)
        if env.use_gui:
            traci.vehicle.setColor("veh2", (178, 102, 255))

        # Overtaking case
        traci.vehicle.moveTo("veh0", "highway_0", s0)
        traci.vehicle.moveTo("veh1", "highway_0", s0 + 50)
        traci.vehicle.moveTo("veh2", "highway_1", s0 - 150)
        traci.vehicle.moveTo("veh3", "highway_2", s0 - 50)
        traci.vehicle.moveTo("veh4", "highway_2", s0 - 0)
        traci.vehicle.setSpeed("veh0", 15)
        traci.vehicle.setSpeed("veh1", 15)
        traci.vehicle.setSpeed("veh2", 55)
        traci.vehicle.setSpeed("veh3", 15)
        traci.vehicle.setSpeed("veh4", 15)
        if save_video:
            video_folder = f"../videos/fast_overtaking___{name}__{label}__{filepath[8:-1]}"
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom("View #0", 10500)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        observation, reward, done, info = env.step(0)
        if env.use_gui:
            traci.vehicle.setColor("veh2", (178, 102, 255))
        for veh in env.vehicles[1:]:
            traci.vehicle.setSpeed(veh, -1)

        # Run fast overtaking case
        action_log = []
        q_log = []
        cv_log = []
        v_log = []
        unc = []
        nb_safe_actions = 0
        for i in range(8):
            if save_video:
                traci.gui.screenshot(
                    "View #0", video_folder + "/images/" + str(i + 1) + ".png"
                )
            action, action_info = dqn.forward(observation)
            observation, reward, done, info = env.step(action, {})
            if env.use_gui:
                traci.vehicle.setColor("veh2", (178, 102, 255))
            if 'safe_action' in action_info:
                if action_info['safe_action']:
                    nb_safe_actions += 1
            action_log.append(action)
            q_log.append(action_info['mean'] if p.agent_par['model'] in ["bnn", "ae"] else action_info['q_values'])
            cv_log.append(action_info['coefficient_of_variation'] if p.agent_par['model'] in ["bnn", "ae"] else action_info['q_values']*0)
            unc.append(
                action_info["coefficient_of_variation"][action]
                if p.agent_par["model"] in ["bnn", 'ae']
                else 0
            )
            v_log.append(env.speeds[0, 0])

        f1 = plt.figure(1)
        f1.set_size_inches(7.5, 4.5)
        plt.rc('font', size=14)
        ax1 = plt.gca()
        ax1_ = ax1.twinx()
        cv_log = np.abs(np.array(cv_log))
        ax1.plot(cv_log[:, 0], label='$\dot{v}_{x,0} = 0$')
        ax1.plot(cv_log[:, 1], label='$\dot{v}_{x,0} = 1$')
        ax1.plot(cv_log[:, 2], label='$\dot{v}_{x,0} = -1$')
        ax1.plot(cv_log[:, 3], label='$\dot{v}_{x,0} = -4$')
        ax1.plot(unc, label='action')
        ax1.legend(loc='upper left')
        if p.agent_par["model"] == 'bnn':
            ax1.axis([0, 7, 0, 0.02])
        # else:
        #     ax1.axis([0, 7, 0, 3])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Uncertainty, $c_\mathrm{v}$")
        ax1.axhline(y=np.abs(safety_threshold), color='k', linestyle='--')
        y_height = np.abs(safety_threshold)
        ax1.text(-0.7, y_height, "$c_\mathrm{v}^\mathrm{safe}$", rotation=0)
        if p.agent_par["model"] == 'bnn':
            ax1.set_yticks([0, 0.01, 0.02])
        # else:
        #     ax1.set_yticks([0, 1.5, 3])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        h1_ = ax1_.plot(v_log, color='tab:gray', linestyle='-.', label='$v_{x,0}$')
        ax1_.axis([0, 7, -0.05, 25.05])
        ax1_.set_ylabel("Speed (m/s)")
        ax1_.set_yticks([0, 10, 20])
        ax1_.spines['left'].set_visible(False)
        ax1_.spines['top'].set_visible(False)
        ax1_.legend(loc='upper right')

        plt.tight_layout()
        f1.savefig(f'../videos/{name}-{label}-g1.png')
        # f1.show()
        plt.close("all")

elif case == "standstill":
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema("View #0", "scheme_for_videos")
        except Exception as e:
            print("SUMO scheme not defined.")
    for _ in range(5):  # Display the case 5 times
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)
        env.reset()
        if env.use_gui:
            traci.vehicle.setColor("veh1", (255, 255, 255))
        s0 = 1000.0
        traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
        traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
        traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
        traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
        traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
        traci.vehicle.moveTo("veh5", "highway_2", s0 - 600)
        traci.vehicle.moveTo("veh6", "highway_2", s0 - 700)
        traci.simulationStep()
        env.speeds[0, 0] = 25
        for veh in env.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setLaneChangeMode(veh, 0)  # Turn off all lane changes
        traci.vehicle.setSpeed("veh0", 25)
        traci.vehicle.setSpeed("veh1", 0)
        traci.vehicle.setSpeed("veh2", 15)
        traci.vehicle.setSpeed("veh3", 15)
        traci.vehicle.setSpeed("veh4", 15)
        traci.vehicle.setSpeed("veh5", 15)
        traci.vehicle.setSpeed("veh6", 15)
        traci.simulationStep()
        traci.vehicle.setMaxSpeed("veh0", 25)
        traci.vehicle.setMaxSpeed("veh1", 0.001)
        traci.vehicle.setMaxSpeed("veh2", 15)
        traci.vehicle.setMaxSpeed("veh3", 15)
        traci.vehicle.setMaxSpeed("veh4", 15)
        traci.vehicle.setMaxSpeed("veh5", 15)
        traci.vehicle.setMaxSpeed("veh6", 15)
        traci.simulationStep()
        env.step(0)
        if env.use_gui:
            traci.vehicle.setColor("veh1", (255, 255, 255))

        # Standstill case
        traci.vehicle.moveTo("veh0", "highway_0", s0)
        traci.vehicle.moveTo("veh1", "highway_0", s0 + 300)
        traci.vehicle.moveTo("veh2", "highway_1", s0 + 36)
        traci.vehicle.moveTo("veh3", "highway_1", s0 + 54)
        traci.vehicle.moveTo("veh4", "highway_1", s0 + 72)
        traci.vehicle.moveTo("veh5", "highway_1", s0 + 90)
        traci.vehicle.moveTo("veh6", "highway_1", s0 + 108)
        traci.vehicle.setSpeed("veh0", 25)
        traci.vehicle.setSpeed("veh1", 0)
        traci.vehicle.setSpeed("veh2", 15)
        traci.vehicle.setSpeed("veh3", 15)
        traci.vehicle.setSpeed("veh4", 15)
        traci.vehicle.setSpeed("veh5", 15)
        traci.vehicle.setSpeed("veh6", 15)
        if save_video:
            video_folder = f"../videos/standstill___{name}__{label}__{filepath[8:-1]}"
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom("View #0", 6850)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        observation, reward, done, info = env.step(0)
        if env.use_gui:
            traci.vehicle.setColor("veh1", (255, 255, 255))
        for veh in env.vehicles[1:]:
            traci.vehicle.setSpeed(veh, -1)

        # Run standstill case
        action_log = []
        q_log = []
        cv_log = []
        unc = []
        v_log = []
        nb_safe_actions = 0
        for i in range(15):
            if save_video:
                traci.gui.screenshot(
                    "View #0", video_folder + "/images/" + str(i + 1) + ".png"
                )
            action, action_info = dqn.forward(observation)
            observation, reward, done, info = env.step(action)
            if env.use_gui:
                traci.vehicle.setColor("veh1", (255, 255, 255))
            if "safe_action" in action_info:
                if action_info["safe_action"]:
                    nb_safe_actions += 1
            action_log.append(action)
            q_log.append(
                action_info["mean"]
                if p.agent_par["model"] in ["bnn", 'ae']
                else action_info["q_values"]
            )
            cv_log.append(
                action_info["coefficient_of_variation"]
                if p.agent_par["model"] in ["bnn", 'ae']
                else action_info["q_values"] * 0
            )
            unc.append(
                action_info["coefficient_of_variation"][action]
                if p.agent_par["model"] in ["bnn", 'ae']
                else 0
            )
            v_log.append(env.speeds[0, 0])

        # Plot results
        f0 = plt.figure(0)
        ax0 = plt.gca()
        q_log = np.array(q_log)
        for action in range(0, np.shape(q_log)[1]):
            ax0.plot(q_log[:, action], label=str(action))
        ax0.legend()
        f0.savefig(f'../videos/{name}-{label}-f0.png')
        # f0.show()

        f1 = plt.figure(1)
        f1.set_size_inches(7.5, 4.5)
        plt.rc("font", size=14)
        ax1 = plt.gca()
        ax1_ = ax1.twinx()
        cv_log = np.abs(np.array(cv_log))
        unc = np.abs(np.array(unc))
        ax1.plot(cv_log[:, 0], label="$\dot{v}_{x,0} = 0$")
        ax1.plot(cv_log[:, 1], label="$\dot{v}_{x,0} = 1$")
        ax1.plot(cv_log[:, 2], label="$\dot{v}_{x,0} = -1$")
        ax1.plot(cv_log[:, 3], label="$\dot{v}_{x,0} = -4$")
        ax1.plot(unc, label="action")
        ax1.legend(loc="upper left")
        # if p.agent_par["model"] == 'bnn':
        #     ax1.axis([0, 10, 0, 0.02])
        # else:
        #     ax1.axis([0, 10, 0, 2])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Uncertainty, $c_\mathrm{v}$")
        ax1.axhline(y=np.abs(safety_threshold), color="k", linestyle="--")
        y_height = np.abs(safety_threshold)
        ax1.text(-0.7, y_height, "$c_\mathrm{v}^\mathrm{safe}$", rotation=0)
        # if p.agent_par["model"] == 'bnn':
        #     ax1.set_yticks([0, 0.01, 0.02])
        # else:
        #     ax1.set_yticks([0, 1, 2])
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)

        h1_ = ax1_.plot(v_log, color="tab:gray", linestyle="-.", label="$v_{x,0}$")
        ax1_.axis([0, 10, -0.05, 25.05])
        ax1_.set_ylabel("Speed (m/s)")
        ax1_.set_yticks([0, 10, 20])
        ax1_.spines["left"].set_visible(False)
        ax1_.spines["top"].set_visible(False)
        ax1_.legend(loc="upper right")

        plt.tight_layout()
        f1.savefig(f'../videos/{name}-{label}-f1.png')
        # f1.show()

        f2 = plt.figure(2)
        ax2 = plt.gca()
        ax2.plot(v_log)
        f2.savefig(f'../videos/{name}-{label}-f2.png')
        # f2.show()
        plt.close("all")

else:
    raise Exception("Case not defined.")
