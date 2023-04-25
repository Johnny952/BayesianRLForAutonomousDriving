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
import numpy as np
from rl.policy import GreedyQPolicy
from rl.memory import Memory

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
from base.run_agent_utils import rerun_test_scenarios, fast_overtaking, standstill
from matplotlib import rcParams
from safe_greedy_policy import SafeGreedyPolicy, SimpleSafeGreedyPolicy

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

""" Options: """
filepath = "../logs/train_agent_20230418_225936_bnn_6M_v6/"
agent_name = "5950075"
case = "all"  # 'rerun_test_scenarios', 'fast_overtaking', 'standstill', 'all'
use_safe_action = True

thresh_range = [0, 50]
thresh_steps = 200
""" End options """

safe_action = 3 if use_safe_action else None

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps

ps.sim_params["remove_sumo_warnings"] = False
nb_actions = len(ps.sim_params["action_interp"])

nb_observations = 4 + 4 * ps.sim_params['sensor_nb_vehicles']
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
dqn.compile(p.agent_par["learning_rate"])

dqn.load_weights(filepath + agent_name)
dqn.training = False

def change_thresh_fn(thresh):
    if p.agent_par["model"] == "bnn":
        dqn.test_policy = SafeGreedyPolicy(
            safety_threshold=thresh, safe_action=safe_action
        )
    elif p.agent_par["model"] == "ae":
        dqn.test_policy = SimpleSafeGreedyPolicy(thresh, safe_action)


if case == "rerun_test_scenarios":
    rerun_test_scenarios(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )
elif case == "fast_overtaking":
    fast_overtaking(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )

elif case == "standstill":
    standstill(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )
elif case == "all":
    rerun_test_scenarios(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )
    fast_overtaking(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )
    standstill(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        thresh_steps=thresh_steps,
        use_safe_action=use_safe_action,
    )
else:
    raise Exception("Case not defined.")
