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
from keras.optimizers import Adam
from keras.models import model_from_json, clone_model
from rl.policy import GreedyQPolicy
from rl.memory import Memory
from dqn_ensemble import DQNAgentEnsemble
from policy import EnsembleTestPolicy
from matplotlib import rcParams
sys.path.append("..")
from dqn_mix import MixDQNAgent, MixTestPolicy
from dqn_standard import DQNAgent
from run_agent_utils import rerun_test_scenarios_v3, rerun_test_scenarios_v0

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

""" Options: """
q_filepath = "../logs/train_agent_20230323_235314_dqn_6M_v3/"
q_agent_name = "5950056"

u_filepath = "../logs/train_agent_20230628_172622_rpf_v10/"#"../logs/train_agent_20230323_235219_rpf_6M_v3/", "../logs/train_agent_20230323_235314_dqn_6M_v3/"
u_agent_name = "5950057"#rpf: "5950033", dqn: "5950056"

number_episodes=1000
csv_sufix='_uncerts'
""" End options """

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, q_filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps

nb_actions = len(ps.sim_params["action_interp"])
nb_observations = 4 + 4 * ps.sim_params['sensor_nb_vehicles']

with open(q_filepath + "model.txt") as text_file:
    saved_model = model_from_json(text_file.read())
model = saved_model
print(model.summary())

memory = Memory(
    window_length=p.agent_par["window_length"]
)  # Not used, simply needed to create the agent

q_policy = GreedyQPolicy()
q_dqn = DQNAgent(
    model=model,
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

with open(u_filepath + "model.txt") as text_file:
    saved_model = model_from_json(text_file.read())
if p.agent_par["ensemble"]:
    models = []
    for i in range(p.agent_par["number_of_networks"]):
        models.append(clone_model(saved_model))
    print(models[0].summary())
else:
    model = saved_model
    print(model.summary())

memory = Memory(
    window_length=p.agent_par["window_length"]
)  # Not used, simply needed to create the agent

if p.agent_par["ensemble"]:
    u_policy = GreedyQPolicy()
    u_test_policy = EnsembleTestPolicy("mean")
    u_dqn = DQNAgentEnsemble(
        models=models,
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
    )
else:
    u_policy = GreedyQPolicy()
    u_test_policy = GreedyQPolicy()
    u_dqn = DQNAgent(
        model=model,
        policy=u_policy,
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
u_dqn.compile(Adam(lr=p.agent_par["learning_rate"]))

u_dqn.load_weights(u_filepath + u_agent_name)
u_dqn.training = False



policy = MixTestPolicy(safety_threshold=None, safe_action=3)
dqn = MixDQNAgent(
    q_model=q_dqn,
    u_model=u_dqn,
    policy=policy,
)

rerun_test_scenarios_v0(
    dqn,
    u_filepath,
    ps,
    use_safe_action=False,
    save_video=False,
    do_save_metrics=False,
    number_tests=1,
    use_gui=False,
    number_episodes=number_episodes,
    do_save_uncert=True,
    csv_sufix=csv_sufix,
)
