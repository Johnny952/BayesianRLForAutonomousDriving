import random  # Required to set random seed for replay memory
import os
import datetime
import sys
from shutil import copyfile
import numpy as np

sys.path.append("..")

import base.parameters as p
import base.parameters_simulation as ps

from base.driving_env import Highway
from base.callbacks import SaveWeights

from keras.callbacks import TensorBoard
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

from dqn_ae import DQNAEAgent2
from network_architecture import NetworkMLP, NetworkCNN, NetworkAE


filepath = "../logs/train_agent_20230828_020015_ae_v22/"
agent_name = "5950008"


# Set log path and name
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = (
    os.path.basename(__file__)[0:-3]
    + "_"
    + start_time
    + ("_" + sys.argv[1] if len(sys.argv) > 1 else "")
)
save_path = "../logs/" + log_name

# Save parameters and code
if not os.path.isdir(save_path):
    if not os.path.isdir("../logs"):
        os.mkdir("../logs")
    os.mkdir(save_path)
    os.mkdir(save_path + "/src")
for file in os.listdir("../base"):
    if file[-3:] == ".py":
        copyfile("../base/" + file, save_path + "/src/" + file[:-3] + "_stored.py")
for file in os.listdir("."):
    if file[-3:] == ".py":
        copyfile("./" + file, save_path + "/src/" + file[:-3] + "_stored.py")


env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
nb_actions = env.nb_actions
nb_observations = env.nb_observations

np.random.seed(p.random_seed)
random.seed(p.random_seed)  # memory.py uses random module

save_weights_callback = SaveWeights(p.save_freq, save_path)
tensorboard_callback = TensorBoard(
    log_dir=save_path, histogram_freq=0, write_graph=True, write_images=False
)
callbacks = [tensorboard_callback, save_weights_callback]

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
    min_covar=p.agent_par["min_covar"],
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
policy = GreedyQPolicy()
test_policy = GreedyQPolicy()
memory = SequentialMemory(
    limit=p.agent_par["buffer_size"], window_length=p.agent_par["window_length"]
)
dqn = DQNAEAgent2(
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
    train_interval=1,
    target_model_update=p.agent_par["target_network_update_freq"],
    delta_clip=p.agent_par["delta_clip"],
    device=p.agent_par["device"],
    update_ae_each=1,
)

dqn.compile(p.agent_par["learning_rate"])
dqn.partial_load_weights(filepath + agent_name)
dqn.training = True

# Run training
dqn.fit(
    env, nb_steps=1e6, visualize=False, verbose=2, callbacks=callbacks
)
