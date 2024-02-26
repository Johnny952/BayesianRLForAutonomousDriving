import sys
import os
import numpy as np
from rl.memory import Memory
from rl.policy import GreedyQPolicy
from keras.models import model_from_json
from keras.optimizers import Adam

sys.path.append("..")
from base.run_agent_utils import rerun_test_scenarios_v3
from base.dqn_standard import DQNAgent
from matplotlib import rcParams
from safe_greedy_policy import RandomSafePolicy
from base.dqn_mix import MixDQNAgent, MixTestPolicy

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

debug = False

""" Options: """
q_filepath = "../logs/train_agent_20230323_235314_dqn_6M_v3/"
q_agent_name = "5950056"
save_path = "../logs/random_agent/"

case = "all"  # 'all', 'uncert'

thresh_range = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
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

q_dqn = DQNAgent(
    model=q_model,
    policy=GreedyQPolicy(),
    test_policy=GreedyQPolicy(),
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


policy = RandomSafePolicy(safety_threshold=0, safe_action=safe_action)
dqn = MixDQNAgent(
    q_model=q_dqn,
    u_model=q_dqn,
    policy=policy,
)


def change_thresh_fn(thresh):
    dqn.policy.safety_threshold = thresh

if not os.path.exists(save_path):
    os.makedirs(save_path)

rerun_test_scenarios_v3(
    dqn,
    save_path,
    ps,
    change_thresh_fn=change_thresh_fn,
    thresh_range=thresh_range,
    use_safe_action=True,
    save_video=save_video,
    do_save_metrics=do_save_metrics,
    number_tests=number_tests,
    use_gui=use_gui,
    number_episodes=number_episodes,
    csv_sufix=csv_sufix,
    do_save_uncert=do_save_uncert,
)
