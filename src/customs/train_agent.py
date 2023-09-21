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
from base.callbacks import SaveWeights, EvaluateAgent

from keras.callbacks import TensorBoard
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

from dqn_bnn import DQNBNNAgent
from dqn_ae import DQNAEAgent
from network_architecture import NetworkMLPBNN, NetworkCNNBNN, NetworkMLP, NetworkCNN, NetworkAE, NetworkAESimple



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
evaluate_agent_callback = EvaluateAgent(
    eval_freq=p.eval_freq, nb_eval_eps=p.nb_eval_eps, save_path=save_path
)
tensorboard_callback = TensorBoard(
    log_dir=save_path, histogram_freq=0, write_graph=True, write_images=False
)
callbacks = [tensorboard_callback, save_weights_callback, evaluate_agent_callback]

# This structure initializes the agent. The different options allows the choice of using a
# convolutional or fully connected neural network architecture,
# and to run the backpropagation of the ensemble members in parallel or sequential.
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
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=p.agent_par["exploration_final_eps"],
        value_test=0.0,
        nb_steps=p.agent_par["exploration_steps"],
    )
    test_policy = GreedyQPolicy()
    memory = SequentialMemory(
        limit=p.agent_par["buffer_size"], window_length=p.agent_par["window_length"]
    )
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
        device=p.agent_par["device"],
        complexity_kld_weight=p.agent_par["complexity_kld_weight"],
        sample_forward=p.agent_par["sample_forward"],
        sample_backward=p.agent_par["sample_backward"],
    )
elif p.agent_par["model"] == 'ae':
    act_obs_ae = NetworkAE(
        p.agent_par["window_length"],
        nb_observations,
        actions=ps.sim_params['action_interp'],
        obs_encoder_arc=p.agent_par["act_obs obs_encoder_arc"],
        act_encoder_arc=p.agent_par["act_obs act_encoder_arc"],
        shared_encoder_arc=p.agent_par["act_obs shared_encoder_arc"],
        obs_decoder_arc=p.agent_par["act_obs obs_decoder_arc"],
        act_decoder_arc=p.agent_par["act_obs act_decoder_arc"],
        shared_decoder_arc=p.agent_par["act_obs shared_decoder_arc"],
        covar_decoder_arc=p.agent_par["act_obs covar_decoder_arc"],
        latent_dim=p.agent_par["act_obs latent_dim"],
        act_loss_weight=p.agent_par["act_obs act_loss_weight"],
        obs_loss_weight=p.agent_par["act_obs obs_loss_weight"],
        prob_loss_weight=p.agent_par["act_obs prob_loss_weight"],
    ).to(p.agent_par['device'])
    obs_ae = NetworkAESimple(
        p.agent_par["window_length"],
        nb_observations,
        encoder_arc=p.agent_par["obs encoder_arc"],
        shared_decoder_arc=p.agent_par["obs shared_encoder_arc"],
        decoder_arc=p.agent_par["obs decoder_arc"],
        covar_decoder_arc=p.agent_par["obs covar_decoder_arc"],
        latent_dim=p.agent_par["obs latent_dim"],
        input_loss_weight=p.agent_par["obs input_loss_weight"],
        prob_loss_weight=p.agent_par["obs prob_loss_weight"],
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
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=p.agent_par["exploration_final_eps"],
        value_test=0.0,
        nb_steps=p.agent_par["exploration_steps"],
    )
    test_policy = GreedyQPolicy()
    memory = SequentialMemory(
        limit=p.agent_par["buffer_size"], window_length=p.agent_par["window_length"]
    )
    dqn = DQNAEAgent(
        model,
        act_obs_ae,
        obs_ae,
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

# Run training
dqn.fit(
    env, nb_steps=p.nb_training_steps, visualize=False, verbose=2, callbacks=callbacks
)
