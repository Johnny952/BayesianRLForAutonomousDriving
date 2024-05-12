"""
    Threshold sweep test episodes
"""

import sys
import numpy as np
from rl.policy import GreedyQPolicy
from rl.memory import Memory
import torch

sys.path.append("..")
from dqn_bnn import DQNBNNAgent
from dqn_ae import DQNAEAgent
from network_architecture import (
    NetworkMLPBNN,
    NetworkCNNBNN,
    NetworkAE,
    NetworkMLP,
    NetworkCNN,
)
from base.run_agent_utils import (
    fast_overtaking_v2,
    standstill_v2,
    rerun_test_scenarios_v3,
)
from matplotlib import rcParams
from safe_greedy_policy import SafeGreedyPolicy, SimpleSafeGreedyPolicy, SafeEnsembleTestPolicy
from base.policy import EnsembleTestPolicy
from base.dqn_mix import RPFDAEAgent
from base.memory import BootstrappingMemory
from base.driving_env import Highway

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

test = 2

""" Options: """
filepath = "../logs/train_dae_rpf_agent_20240421_171846/"# train_agent_20231006_154948_dae_v5
agent_name = "5950020"# 5950036
dae_rpf = True
if test == 1:
    case = "all-no-rerun"  # 'rerun_test_scenarios', 'all-no-rerun'
    use_safe_action = False
    do_save_metrics = False
    do_save_uncert = True
elif test == 2:
    case = "rerun_test_scenarios"  # 'rerun_test_scenarios', 'all-no-rerun'
    use_safe_action = True
    do_save_metrics = True
    do_save_uncert = False

thresh_range = [
    99.31904519807995,
    100.0740765429599,
    99.71165875,
    100.0417375,
    100.5090662,
    100.92246787,
    102.3692832340001,
    250,
]

save_video = False
number_tests = 1
number_episodes = 2000
csv_sufix = "_v3"
position_steps = 100
use_gui = False
""" End options """

safe_action = 3

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps

ps.sim_params["remove_sumo_warnings"] = False
nb_actions = len(ps.sim_params["action_interp"])
nb_observations = 4 + 4 * ps.sim_params["sensor_nb_vehicles"]


if dae_rpf:
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
    nb_ego_states = env.nb_ego_states
    nb_states_per_vehicle = env.nb_states_per_vehicle
    env.close()
    nb_models = p.agent_par['number_of_networks']
    policy = GreedyQPolicy()
    if use_safe_action:
        test_policy = SafeEnsembleTestPolicy('mean', 0, safe_action)
    else:
        test_policy = EnsembleTestPolicy('mean')
    memory = BootstrappingMemory(nb_nets=p.agent_par['number_of_networks'], limit=p.agent_par['buffer_size'],
                                    adding_prob=p.agent_par["adding_prob"], window_length=p.agent_par["window_length"])
    dqn = RPFDAEAgent(update_ae_each=p.agent_par['update_ae_each'], nb_models=nb_models, learning_rate=p.agent_par['learning_rate'],
                                    nb_ego_states=nb_ego_states, nb_states_per_vehicle=nb_states_per_vehicle,
                                    nb_vehicles=ps.sim_params['sensor_nb_vehicles'],
                                    nb_conv_layers=p.agent_par['nb_conv_layers'],
                                    nb_conv_filters=p.agent_par['nb_conv_filters'],
                                    nb_hidden_fc_layers=p.agent_par['nb_hidden_fc_layers'],
                                    nb_hidden_neurons=p.agent_par['nb_hidden_neurons'], policy=policy,
                                    test_policy=test_policy, enable_double_dqn=p.agent_par['double_q'],
                                    enable_dueling_network=False, nb_actions=nb_actions,
                                    prior_scale_factor=p.agent_par['prior_scale_factor'],
                                    window_length=p.agent_par['window_length'], memory=memory,
                                    gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                                    nb_steps_warmup=p.agent_par['learning_starts'],
                                    train_interval=p.agent_par['train_freq'],
                                    target_model_update=p.agent_par['target_network_update_freq'],
                                    delta_clip=p.agent_par['delta_clip'], network_seed=p.random_seed)

    dae = NetworkAE(
        p.agent_par["window_length"],
        nb_observations,
        actions=ps.sim_params['action_interp'],
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
        min_value=p.agent_par["min_value"],
    ).to(p.agent_par['device'])

    dae_optimizer = torch.optim.Adam(
        dae.parameters(), lr=p.agent_par["learning_rate"]
    )
    dqn.set_uncertainty_model(dae, dae_optimizer, p.agent_par['device'])
else:
    if p.agent_par["model"] == "bnn":
        if p.agent_par["cnn"]:
            model = NetworkCNNBNN(
                4,
                4,
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
        safety_threshold_ = 0 if use_safe_action else None
        if use_safe_action:
            test_policy = SafeGreedyPolicy(
                safety_threshold=safety_threshold_, safe_action=safe_action
            )
        else:
            test_policy = GreedyQPolicy()
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
            device=p.agent_par["device"],
        )
        dqn.set_models()
    elif p.agent_par["model"] == "ae":
        ae = NetworkAE(
            p.agent_par["window_length"],
            nb_observations,
            actions=ps.sim_params['action_interp'],
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
            min_value=p.agent_par["min_value"],
        ).to(p.agent_par["device"])
        if p.agent_par["cnn"]:
            model = NetworkCNN(
                4,
                4,
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
            test_policy = SimpleSafeGreedyPolicy(0, safe_action)
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
            update_ae_each=p.agent_par["update_ae_each"],
        )
    else:
        raise Exception("Model not implemented.")
try:
    dqn.compile(p.agent_par["learning_rate"])
except NotImplementedError:
    print('Model not compiled')

dqn.load_weights(filepath + agent_name)
dqn.training = False


def change_thresh_fn(thresh):
    if dae_rpf:
        dqn.test_policy = SafeEnsembleTestPolicy('mean', thresh, safe_action)
    else:
        if p.agent_par["model"] == "bnn":
            dqn.test_policy = SafeGreedyPolicy(
                safety_threshold=thresh, safe_action=safe_action
            )
        elif p.agent_par["model"] == "ae":
            dqn.test_policy = SimpleSafeGreedyPolicy(thresh, safe_action)


if case == "rerun_test_scenarios":
    rerun_test_scenarios_v3(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        use_safe_action=use_safe_action,
        save_video=save_video,
        do_save_metrics=do_save_metrics,
        number_tests=number_tests,
        use_gui=use_gui,
        number_episodes=number_episodes,
        csv_sufix=csv_sufix,
        do_save_uncert=do_save_uncert,
    )
elif case == "all-no-rerun":
    fast_overtaking_v2(
        dqn,
        filepath,
        ps,
        use_safe_action=use_safe_action,
        save_video=save_video,
        position_steps=position_steps,
        use_gui=use_gui,
        csv_sufix=csv_sufix,
        do_save_uncert=True,
    )
    standstill_v2(
        dqn,
        filepath,
        ps,
        use_safe_action=use_safe_action,
        save_video=save_video,
        position_steps=position_steps,
        use_gui=use_gui,
        csv_sufix=csv_sufix,
        do_save_uncert=True,
    )
else:
    raise Exception("Case not defined.")
