import sys
import numpy as np
from rl.policy import GreedyQPolicy
from rl.memory import Memory
from keras.models import model_from_json
from keras.optimizers import Adam

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
from base.run_agent_utils import rerun_test_scenarios_v3, rerun_test_scenarios_v0
from base.dqn_mix import MixDQNAgent, MixTestPolicy, MixEWMATestPolicy
from base.dqn_standard import DQNAgent
from matplotlib import rcParams
from safe_greedy_policy import SafeGreedyPolicy, SimpleSafeGreedyPolicy, SimpleSafeGreedyPolicyHard

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

debug = False

""" Options: """
q_filepath = "../logs/train_agent_20230323_235314_dqn_6M_v3/"
q_agent_name = "5950056"

u_filepath = "../logs/train_agent_20231006_154948_dae_v5/"
u_agent_name = "5950036"
use_safe_action = True

case = "all"  # 'all', 'uncert'

thresh_range = [
    -23.598912086897165,
-7.6908720583329035,
-13.175728,
-5.6123715,
6.165326525000014,
19.04202440000001,
81.67095490000519,
    100000,
]
if debug:
    save_video = True
    do_save_metrics = False
    number_episodes = 50
    use_gui = True
else:
    save_video = False
    do_save_metrics = True
    number_episodes = 2000
    use_gui = False

do_save_uncert = False
number_tests = 1
csv_sufix = "_v5"

""" End options """

safe_action = 3


# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, q_filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps

nb_actions = len(ps.sim_params["action_interp"])
nb_observations = 4 + 4 * ps.sim_params["sensor_nb_vehicles"]

with open(q_filepath + "model.txt") as text_file:
    saved_model = model_from_json(text_file.read())
q_model = saved_model
print(q_model.summary())

memory = Memory(
    window_length=p.agent_par["window_length"]
)  # Not used, simply needed to create the agent

q_policy = GreedyQPolicy()
q_dqn = DQNAgent(
    model=q_model,
    policy=q_policy,
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
)
q_dqn.compile(Adam(lr=p.agent_par["learning_rate"]))

q_dqn.load_weights(q_filepath + q_agent_name)
q_dqn.training = False


# These import statements need to come after the choice of which agent that should be used.
sys.path.pop(0)
sys.path.insert(0, u_filepath)
import src.parameters_stored as p
import src.parameters_simulation_stored as ps

ps.sim_params["remove_sumo_warnings"] = False

if p.agent_par["model"] == "bnn":
    if p.agent_par["cnn"]:
        u_model = NetworkCNNBNN(
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
        u_model = NetworkMLPBNN(
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

    u_policy = GreedyQPolicy()
    safety_threshold_ = 0 if use_safe_action else None
    if use_safe_action:
        u_test_policy = SafeGreedyPolicy(
            safety_threshold=safety_threshold_, safe_action=safe_action
        )
    else:
        u_test_policy = GreedyQPolicy()
    u_dqn = DQNBNNAgent(
        model=u_model,
        policy=u_policy,
        test_policy=u_test_policy,
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
    u_dqn.set_models()
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
        u_model = NetworkCNN(
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
        u_model = NetworkMLP(
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

    u_policy = GreedyQPolicy()
    if use_safe_action:
        u_test_policy = SimpleSafeGreedyPolicy(0, safe_action)
    else:
        u_test_policy = GreedyQPolicy()
    u_dqn = DQNAEAgent(
        u_model,
        ae,
        u_policy,
        u_test_policy,
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
u_dqn.compile(p.agent_par["learning_rate"])

u_dqn.load_weights(u_filepath + u_agent_name)
u_dqn.training = False


policy = MixTestPolicy(safety_threshold=None, safe_action=3)
# policy = MixEWMATestPolicy(alpha=0.075, safe_action=3, offset=None)
dqn = MixDQNAgent(
    q_model=q_dqn,
    u_model=u_dqn,
    policy=policy,
)


def change_thresh_fn(thresh):
    dqn.policy.safety_threshold = thresh


if case == "all":
    rerun_test_scenarios_v3(
        dqn,
        u_filepath,
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
    # rerun_test_scenarios_v0(
    #     dqn,
    #     u_filepath,
    #     ps,
    #     change_thresh_fn=change_thresh_fn,
    #     thresh_range=thresh_range,
    #     use_safe_action=True,
    #     save_video=save_video,
    #     do_save_metrics=do_save_metrics,
    #     number_tests=number_tests,
    #     use_gui=use_gui,
    #     number_episodes=number_episodes,
    #     do_save_uncert=do_save_uncert,
    # )
elif case == "uncert":
    rerun_test_scenarios_v3(
        dqn,
        u_filepath,
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
    rerun_test_scenarios_v0(
        dqn,
        u_filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        use_safe_action=use_safe_action,
        save_video=save_video,
        do_save_metrics=do_save_metrics,
        number_tests=number_tests,
        use_gui=use_gui,
        number_episodes=number_episodes,
        do_save_uncert=do_save_uncert,
    )
else:
    raise Exception("Case not defined.")
