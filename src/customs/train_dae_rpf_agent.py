""" 
    BASIC AGENT TRAINING for DAE model to uncertainty estimation and RPF for decision making 
"""
import random  # Required to set random seed for replay memory
import os
import datetime
import sys
from shutil import copyfile
import numpy as np
import torch

sys.path.append("..")

import base.parameters as p
import base.parameters_simulation as ps

from base.driving_env import Highway
from base.callbacks import SaveWeights, EvaluateAgent
from base.dqn_standard import DQNAgent
from base.dqn_ensemble import DQNAgentEnsemble, DQNAgentEnsembleParallel, UpdateActiveModelCallback
from base.memory import BootstrappingMemory
from base.policy import EnsembleTestPolicy
from base.network_architecture import NetworkMLP, NetworkCNN

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from base.dqn_mix import RPFDAEAgent
from customs.network_architecture import NetworkAE

# Set log path and name
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = os.path.basename(__file__)[0:-3]+"_"+start_time+("_"+sys.argv[1] if len(sys.argv) > 1 else "")
save_path = "../logs/"+log_name

# Save parameters and code
if not os.path.isdir(save_path):
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    os.mkdir(save_path)
    os.mkdir(save_path + '/src')
for file in os.listdir('.'):
    if file[-3:] == '.py':
        copyfile('./' + file, save_path + '/src/' + file[:-3] + '_stored.py')

env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
nb_actions = env.nb_actions
nb_observations = env.nb_observations

np.random.seed(p.random_seed)
random.seed(p.random_seed)   # memory.py uses random module

save_weights_callback = SaveWeights(p.save_freq, save_path)
evaluate_agent_callback = EvaluateAgent(eval_freq=p.eval_freq, nb_eval_eps=p.nb_eval_eps, save_path=save_path)
tensorboard_callback = TensorBoard(log_dir=save_path, histogram_freq=0, write_graph=True, write_images=False)
callbacks = [tensorboard_callback, save_weights_callback, evaluate_agent_callback]

# This structure initializes the agent. The different options allows the choice of using a
# convolutional or fully connected neural network architecture,
# and to run the backpropagation of the ensemble members in parallel or sequential.

nb_models = p.agent_par['number_of_networks']
policy = GreedyQPolicy()
test_policy = EnsembleTestPolicy('mean')
memory = BootstrappingMemory(nb_nets=p.agent_par['number_of_networks'], limit=p.agent_par['buffer_size'],
                                adding_prob=p.agent_par["adding_prob"], window_length=p.agent_par["window_length"])
dqn = RPFDAEAgent(nb_models=nb_models, learning_rate=p.agent_par['learning_rate'],
                                nb_ego_states=env.nb_ego_states, nb_states_per_vehicle=env.nb_states_per_vehicle,
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
callbacks.append(UpdateActiveModelCallback(dqn))
model_as_string = dqn.get_model_as_string()



with open(save_path+"/"+'model.txt', 'w') as text_file:
    text_file.write(model_as_string)


############################
# DAE model
############################
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

# Run training
dqn.fit(env, nb_steps=p.nb_training_steps, visualize=False, verbose=2, callbacks=callbacks)
