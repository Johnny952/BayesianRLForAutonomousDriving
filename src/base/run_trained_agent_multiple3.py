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
from matplotlib import rcParams
sys.path.append("..")
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json, clone_model
from rl.policy import GreedyQPolicy
from rl.memory import Memory
from dqn_standard import DQNAgent
from dqn_ensemble import DQNAgentEnsemble
from policy import EnsembleTestPolicy

from run_agent_utils import rerun_test_scenarios_v3, fast_overtaking_v2, standstill_v2

rcParams["pdf.fonttype"] = 42  # To avoid Type 3 fonts in figures
rcParams["ps.fonttype"] = 42

""" Options: """
filepath = "../logs/train_agent_20230715_211722_rpf_v14/"#"../logs/train_agent_20230715_211722_rpf_v14/", "../logs/train_agent_20230323_235314_dqn_6M_v3/"
agent_name = "5950023"#rpf: "5950033", dqn: "5950056"
case = "rerun_test_scenarios"  # 'rerun_test_scenarios', 'all-no-rerun'
use_ensemble_test_policy = True

thresh_range = [
    0.018733895607714914,
    0.02182466209477983,
    0.01993404775,
    0.021279978449999997,
    0.023367345289999997,
    0.02548624245000003,
    0.03320669617200001,
    0.15390997,
]
save_video = False
do_save_metrics = True
do_save_uncert = False
number_tests = 1
number_episodes=2000
csv_sufix='_v3'
position_steps=100
use_gui=False
""" End options """

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + "src/")
import parameters_stored as p
import parameters_simulation_stored as ps


ps.sim_params["remove_sumo_warnings"] = False
nb_actions = len(ps.sim_params["action_interp"])

nb_observations = 4 + 4 * ps.sim_params['sensor_nb_vehicles']

with open(filepath + "model.txt") as text_file:
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
    policy = GreedyQPolicy()
    test_policy = EnsembleTestPolicy("mean")
    dqn = DQNAgentEnsemble(
        models=models,
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
    )
else:
    policy = GreedyQPolicy()
    test_policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        policy=policy,
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
dqn.compile(Adam(lr=p.agent_par["learning_rate"]))

dqn.load_weights(filepath + agent_name)
dqn.training = False

if use_ensemble_test_policy:
    test_policy_safe = EnsembleTestPolicy(safety_threshold=0, safe_action=3)
    dqn.test_policy = test_policy_safe


def change_thresh_fn(thresh):
    dqn.test_policy = EnsembleTestPolicy(safety_threshold=thresh, safe_action=3)


if case == "rerun_test_scenarios":
    rerun_test_scenarios_v3(
        dqn,
        filepath,
        ps,
        change_thresh_fn=change_thresh_fn,
        thresh_range=thresh_range,
        use_safe_action=use_ensemble_test_policy,
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
        use_safe_action=use_ensemble_test_policy,
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
        use_safe_action=use_ensemble_test_policy,
        save_video=save_video,
        position_steps=position_steps,
        use_gui=use_gui,
        csv_sufix=csv_sufix,
        do_save_uncert=True,
    )
else:
    raise Exception("Case not defined.")
