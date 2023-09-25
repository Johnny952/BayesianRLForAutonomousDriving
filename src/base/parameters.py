"""
Parameters of the agent.

The meaning of the different parameters are described below.
Setting debug_run = True allows a shorter run, only for debugging purposes.
"""

debug_run = False
env_seed = 13
random_seed = env_seed+1

nb_training_steps = int(6e6) if not debug_run else int(2e5)   # Number of training steps
save_freq = 50000 if not debug_run else 1000   # Number of training steps between saving the network weights
eval_freq = 50000 if not debug_run else 1000   # Number of training steps between evaluating the agent on test episodes
nb_eval_eps = 100 if not debug_run else 5   # Number of test episodedes

agent_par = {}
agent_par["ensemble"] = True   # Ensemble RPF or standard DQN agent
agent_par["parallel"] = True   # Parallel execution of backpropagation for ensemble RPF
agent_par["number_of_networks"] = 10 if not debug_run else 5   # Number of ensemble members
agent_par["prior_scale_factor"] = 50.   # Prior scale factor, beta
agent_par["adding_prob"] = 0.5   # Probability of adding an experience to each individual ensemble replay memory
agent_par["cnn"] = True   # CNN or MLP
agent_par["nb_conv_layers"] = 2
agent_par["nb_conv_filters"] = 32
agent_par["nb_hidden_fc_layers"] = 2
agent_par["nb_hidden_neurons"] = 64

agent_par["gamma"] = 0.99   # Discount factor
agent_par["learning_rate"] = 0.0005
agent_par["buffer_size"] = 500000 if not debug_run else 5000
agent_par["exploration_steps"] = 1000000   # Steps to anneal the exploration rate to minimum.
agent_par["exploration_final_eps"] = 0.05
agent_par["train_freq"] = 1
agent_par["batch_size"] = 32
agent_par["double_q"] = True   # Use double DQN
agent_par["learning_starts"] = 50000 if not debug_run else 1000   # No training during initial steps
agent_par["target_network_update_freq"] = 20000 if not debug_run else 500
agent_par['duel_q'] = True   # Dueling neural network architecture
agent_par['delta_clip'] = 10.   # Huber loss parameter
agent_par["window_length"] = 1   # How many historic states to include (1 uses only current state)
agent_par["tensorboard_log"] = "../logs/"



agent_par["device"] = 'cuda'
agent_par["model"] = 'ae' # bnn or ae


############## BNN ###############
agent_par["prior_mu"] = 0
agent_par["prior_sigma"] = 0.1
agent_par["complexity_kld_weight"] = 1
agent_par["sample_forward"] = 10
agent_par["sample_backward"] = 1


############### DAE ###############
agent_par["act_obs obs_encoder_arc"] = [64]
agent_par["act_obs act_encoder_arc"] = [32]
agent_par["act_obs shared_encoder_arc"] = [32, 16]
agent_par["act_obs obs_decoder_arc"] = [64]
agent_par["act_obs act_decoder_arc"] = [32]
agent_par["act_obs shared_decoder_arc"] = [16, 32]
agent_par["act_obs covar_decoder_arc"] = [64, 1024]
agent_par["act_obs latent_dim"] = 16
agent_par["act_obs act_loss_weight"] = 1
agent_par["act_obs obs_loss_weight"] = 1
agent_par["act_obs prob_loss_weight"] = 1 / 3000

agent_par["obs encoder_arc"] = [64]
agent_par["obs shared_encoder_arc"] = [32, 16]
agent_par["obs decoder_arc"] = [16, 32]
agent_par["obs covar_decoder_arc"] = [64, 1024]
agent_par["obs latent_dim"] = 16
agent_par["obs input_loss_weight"] = 1
agent_par["obs prob_loss_weight"] = 1 / 3000

agent_par["update_ae_each"] = 50
agent_par["min_value"] = 0.2
