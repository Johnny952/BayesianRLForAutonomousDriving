import torch
import torch.nn as nn
import torchbnn as bnn

def printState(state):
    print(state.shape)
    return state


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor):
        return self.lambd(x)


class AddLayer(nn.Module):
    def __init__(self, model1, model2):
        super(AddLayer, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        return self.model1(x) + self.model2(x)


class MergeLayer(nn.Module):
    def __init__(self, model1, model2):
        super(MergeLayer, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return torch.cat((out1, out2), dim=-1)


class NetworkMLPBNN(nn.Module):
    def __init__(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        duel,
        prior,
        prior_scale_factor=10.0,
        duel_type="avg",
        activation="relu",
        window_length=1,
        prior_mu=0,
        prior_sigma=0.1,
    ):
        super(NetworkMLPBNN, self).__init__()
        self.model = None
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        if not prior and not duel:
            self.build_mlp(
                nb_inputs,
                nb_outputs,
                nb_hidden_layers,
                nb_hidden_neurons,
                activation=activation,
                window_length=window_length,
            )
        elif not prior and duel:
            self.build_mlp_dueling(
                nb_inputs,
                nb_outputs,
                nb_hidden_layers,
                nb_hidden_neurons,
                dueling_type=duel_type,
                activation=activation,
                window_length=window_length,
            )
        elif prior and not duel:
            self.build_prior_plus_trainable(
                nb_inputs,
                nb_outputs,
                nb_hidden_layers,
                nb_hidden_neurons,
                activation=activation,
                prior_scale_factor=prior_scale_factor,
                window_length=window_length,
            )
        elif prior and duel:
            self.build_prior_plus_trainable_dueling(
                nb_inputs,
                nb_outputs,
                nb_hidden_layers,
                nb_hidden_neurons,
                dueling_type=duel_type,
                activation=activation,
                prior_scale_factor=prior_scale_factor,
                window_length=window_length,
            )
        else:
            raise Exception("Error in Network creation")
        
    def predict_on_batch(self, x):
        raise Exception('Not Implemented yet')
    
    def layer(self, in_features, out_features):
        return bnn.BayesLinear(
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x):
        return self.model(x)

    def build_mlp_layer(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
    ):
        act = None
        if activation == "relu":
            act = nn.ReLU
        layers = [
            nn.Flatten(),
            nn.Linear((window_length * nb_inputs), nb_hidden_neurons),
            act(),
        ]

        for _ in range(nb_hidden_layers - 1):
            layers.append(nn.Linear(nb_hidden_neurons, nb_hidden_neurons))
            layers.append(act())
        layers.append(nn.Linear(nb_hidden_neurons, nb_outputs))
        return layers

    def build_bnn_mlp_layer(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
    ):
        act = None
        if activation == "relu":
            act = nn.ReLU
        layers = [
            nn.Flatten(),
            self.layer(
                (window_length * nb_inputs),
                nb_hidden_neurons,
            ),
            act(),
        ]
        for _ in range(nb_hidden_layers - 1):
            layers.append(
                self.layer(
                    nb_hidden_neurons,
                    nb_hidden_neurons,
                ),
            )
            layers.append(act())
        layers.append(
            self.layer(
                nb_hidden_neurons,
                nb_outputs,
            ),
        )
        return layers

    def build_mlp(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
    ):
        layers = self.build_bnn_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        self.model = nn.Sequential(*layers)

    def build_mlp_dueling(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        dueling_type="avg",
        activation="relu",
        window_length=1,
    ):
        layers = self.build_bnn_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        layers = layers[:-1]
        layers.append(
            self.layer(
                nb_hidden_neurons,
                nb_outputs + 1,
            ),
        )
        if dueling_type == "avg":
            layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            layers.append(LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:]))
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = nn.Sequential(*layers)

    def build_prior_plus_trainable(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        activation="relu",
        prior_scale_factor=1.0,
        window_length=1,
    ):

        prior_layers = self.build_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        prior_layers.append(LambdaLayer(lambda x: x * prior_scale_factor))
        prior = nn.Sequential(*prior_layers)
        for param in prior.parameters():
            param.requires_grad = False

        trainable_layers = self.build_bnn_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        trainable_model = nn.Sequential(*trainable_layers)

        self.model = AddLayer(prior, trainable_model)

    def build_prior_plus_trainable_dueling(
        self,
        nb_inputs,
        nb_outputs,
        nb_hidden_layers,
        nb_hidden_neurons,
        activation="relu",
        prior_scale_factor=1.0,
        dueling_type="avg",
        window_length=1,
    ):
        prior_layers = self.build_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )

        prior_layers = prior_layers[:-1]
        prior_layers.append(
            self.layer(
                nb_hidden_neurons,
                nb_outputs + 1,
            ),
        )
        if dueling_type == "avg":
            prior_layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            prior_layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            prior_layers.append(LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:]))
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        prior_layers.append(LambdaLayer(lambda x: x * prior_scale_factor))
        prior = nn.Sequential(*prior_layers)
        for param in prior.parameters():
            param.requires_grad = False

        trainable_layers = self.build_bnn_mlp_layer(
            nb_inputs,
            nb_outputs,
            nb_hidden_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        trainable_layers = trainable_layers[:-1]
        trainable_layers.append(
            self.layer(
                nb_hidden_neurons,
                nb_outputs + 1,
            ),
        )
        if dueling_type == "avg":
            trainable_layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            trainable_layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            trainable_layers.append(
                LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:])
            )
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        trainable_layers.append(LambdaLayer(lambda x: x * prior_scale_factor))
        trainable_model = nn.Sequential(*trainable_layers)

        self.model = AddLayer(prior, trainable_model)


class NetworkCNNBNN(nn.Module):
    def __init__(
        self,
        nb_ego_states,
        nb_states_per_vehicle,
        nb_vehicles,
        nb_actions,
        nb_conv_layers,
        nb_conv_filters,
        nb_hidden_fc_layers,
        nb_hidden_neurons,
        duel,
        prior,
        prior_scale_factor=10.0,
        duel_type="avg",
        activation="relu",
        window_length=1,
        prior_mu=0,
        prior_sigma=0.1,
    ):
        super(NetworkCNNBNN, self).__init__()
        self.model = None
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        if not prior and not duel:
            self.build_cnn(
                nb_ego_states,
                nb_states_per_vehicle,
                nb_vehicles,
                nb_actions,
                nb_conv_layers,
                nb_conv_filters,
                nb_hidden_fc_layers,
                nb_hidden_neurons,
                activation=activation,
                window_length=window_length,
            )
        elif not prior and duel:
            self.build_cnn_dueling(
                nb_ego_states,
                nb_states_per_vehicle,
                nb_vehicles,
                nb_actions,
                nb_conv_layers,
                nb_conv_filters,
                nb_hidden_fc_layers,
                nb_hidden_neurons,
                dueling_type=duel_type,
                activation=activation,
                window_length=window_length,
            )
        elif prior and duel:
            self.build_cnn_dueling_prior(
                nb_ego_states,
                nb_states_per_vehicle,
                nb_vehicles,
                nb_actions,
                nb_conv_layers,
                nb_conv_filters,
                nb_hidden_fc_layers,
                nb_hidden_neurons,
                dueling_type=duel_type,
                activation=activation,
                prior_scale_factor=prior_scale_factor,
                window_length=window_length,
            )
        else:
            raise Exception("Error in Network creation")

    def layer(self, in_features, out_features):
        return bnn.BayesLinear(
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            in_features=in_features,
            out_features=out_features,
        )
    
    def forward(self, x):
        return self.model(x)

    def predict_on_batch(self, x):
        raise Exception('Not Implemented yet')

    def build_cnn_layers(
        self,
        nb_ego_states,
        nb_states_per_vehicle,
        nb_vehicles,
        nb_conv_layers,
        nb_conv_filters,
        nb_hidden_fc_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
    ):
        act = None
        if activation == "relu":
            act = nn.ReLU

        input_ego = nn.Sequential(
            nn.Flatten(),
            LambdaLayer(lambda state: state[:, : nb_ego_states * window_length]),
        )
        input_others_layers = [
            nn.Flatten(),
            LambdaLayer(
                lambda state: state[:, nb_ego_states * window_length :].unsqueeze(dim=1)
            ),
            nn.Conv1d(
                in_channels=1,
                out_channels=nb_conv_filters,
                kernel_size=(nb_states_per_vehicle * window_length),
                stride=(nb_states_per_vehicle * window_length),
            ),
            act(),
        ]
        for _ in range(nb_conv_layers - 1):
            input_others_layers += [
                nn.Conv1d(
                    in_channels=nb_conv_filters,
                    out_channels=nb_conv_filters,
                    kernel_size=1,
                    stride=1,
                ),
                act(),
            ]
        input_others_layers += [
            nn.MaxPool1d(
                kernel_size=nb_vehicles,
            ),
            nn.Flatten(),
        ]
        input_others = nn.Sequential(*input_others_layers)

        output = [
            MergeLayer(input_ego, input_others),
            self.layer(
                (nb_conv_filters + nb_ego_states * window_length),
                nb_hidden_neurons,
            ),
            act(),
        ]
        for _ in range(nb_hidden_fc_layers - 1):
            output += [
                self.layer(
                    nb_hidden_neurons,
                    nb_hidden_neurons,
                ),
                act(),
            ]
        return output

    def build_cnn(
        self,
        nb_ego_states,
        nb_states_per_vehicle,
        nb_vehicles,
        nb_actions,
        nb_conv_layers,
        nb_conv_filters,
        nb_hidden_fc_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
    ):
        output = self.build_cnn_layers(
            nb_ego_states,
            nb_states_per_vehicle,
            nb_vehicles,
            nb_conv_layers,
            nb_conv_filters,
            nb_hidden_fc_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        output.append(
            self.layer(
                nb_hidden_neurons,
                nb_actions,
            ),
        )
        self.model = nn.Sequential(*output)

    def build_cnn_dueling(
        self,
        nb_ego_states,
        nb_states_per_vehicle,
        nb_vehicles,
        nb_actions,
        nb_conv_layers,
        nb_conv_filters,
        nb_hidden_fc_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
        dueling_type="avg",
    ):
        layers = self.build_cnn_layers(
            nb_ego_states,
            nb_states_per_vehicle,
            nb_vehicles,
            nb_conv_layers,
            nb_conv_filters,
            nb_hidden_fc_layers,
            nb_hidden_neurons,
            activation=activation,
            window_length=window_length,
        )
        layers.append(
            self.layer(
                nb_hidden_neurons,
                nb_actions + 1,
            ),
        )

        if dueling_type == "avg":
            layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            layers.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            layers.append(LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:]))
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = nn.Sequential(*layers)

    def build_cnn_dueling_prior(
        self,
        nb_ego_states,
        nb_states_per_vehicle,
        nb_vehicles,
        nb_actions,
        nb_conv_layers,
        nb_conv_filters,
        nb_hidden_fc_layers,
        nb_hidden_neurons,
        activation="relu",
        window_length=1,
        dueling_type="avg",
        prior_scale_factor=1.0,
    ):
        act = None
        if activation == "relu":
            act = nn.ReLU

        input_ego = nn.Sequential(
            nn.Flatten(),
            LambdaLayer(lambda state: state[:, : nb_ego_states * window_length]),
        )
        input_others_reshaped = nn.Sequential(
            nn.Flatten(),
            LambdaLayer(
                lambda state: state[:, nb_ego_states * window_length :].unsqueeze(dim=1)
            ),
        )
        ############ Prior #################
        input_others_layers = [
            input_others_reshaped,
            nn.Conv1d(
                in_channels=1,
                out_channels=nb_conv_filters,
                kernel_size=(nb_states_per_vehicle * window_length),
                stride=(nb_states_per_vehicle * window_length),
            ),
            act(),
        ]
        for _ in range(nb_conv_layers - 1):
            input_others_layers += [
                nn.Conv1d(
                    in_channels=nb_conv_filters,
                    out_channels=nb_conv_filters,
                    kernel_size=1,
                    stride=1,
                ),
                act(),
            ]
        input_others_layers += [
            nn.MaxPool1d(
                kernel_size=nb_vehicles,
            ),
            nn.Flatten(),
        ]
        input_others = nn.Sequential(*input_others_layers)
        prior_merged = [
            MergeLayer(input_ego, input_others),
            self.layer(
                (nb_conv_filters + nb_ego_states * window_length),
                nb_hidden_neurons,
            ),
            act(),
        ]
        for _ in range(nb_hidden_fc_layers - 1):
            prior_merged += [
                self.layer(
                    nb_hidden_neurons,
                    nb_hidden_neurons,
                ),
                act(),
            ]
        prior_merged += [
            self.layer(
                nb_hidden_neurons,
                nb_actions + 1,
            ),
            nn.Flatten(),
        ]
        if dueling_type == "avg":
            prior_merged.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            prior_merged.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            prior_merged.append(LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:]))
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        prior_merged.append(LambdaLayer(lambda x: x * prior_scale_factor))
        prior = nn.Sequential(*prior_merged)
        for param in prior.parameters():
            param.requires_grad = False

        ############ Trainable #################
        trainable_conv_net = [
            input_others_reshaped,
            nn.Conv1d(
                in_channels=1,
                out_channels=nb_conv_filters,
                kernel_size=(nb_states_per_vehicle * window_length),
                stride=(nb_states_per_vehicle * window_length),
            ),
            act(),
        ]
        for _ in range(nb_conv_layers - 1):
            trainable_conv_net += [
                nn.Conv1d(
                    in_channels=nb_conv_filters,
                    out_channels=nb_conv_filters,
                    kernel_size=1,
                    stride=1,
                ),
                act(),
            ]
        trainable_conv_net += [
            nn.MaxPool1d(
                kernel_size=nb_vehicles,
            ),
            nn.Flatten(),
        ]
        trainable_net = nn.Sequential(*trainable_conv_net)

        trainable_merged = [
            MergeLayer(input_ego, trainable_net),
            self.layer(
                (nb_conv_filters + nb_ego_states * window_length),
                nb_hidden_neurons,
            ),
            act(),
        ]
        for _ in range(nb_hidden_fc_layers - 1):
            trainable_merged += [
                self.layer(
                    nb_hidden_neurons,
                    nb_hidden_neurons,
                ),
                act(),
            ]
        trainable_merged += [
            self.layer(
                nb_hidden_neurons,
                nb_actions + 1,
            ),
        ]
        if dueling_type == "avg":
            trainable_merged.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].mean(dim=-1).unsqueeze(-1)
                )
            )
        elif dueling_type == "max":
            trainable_merged.append(
                LambdaLayer(
                    lambda a: a[:, 0].unsqueeze(-1)
                    + a[:, 1:]
                    - a[:, 1:].max(dim=-1).values.unsqueeze(-1)
                )
            )
        elif dueling_type == "naive":
            trainable_merged.append(
                LambdaLayer(lambda a: a[:, 0].unsqueeze(-1) + a[:, 1:])
            )
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        trainable_net = nn.Sequential(*trainable_merged)

        self.model = AddLayer(trainable_net, prior)



########### AE #####################
class NetworkMLP(NetworkMLPBNN):
    def layer(self, in_features, out_features):
        return nn.Linear(
            in_features,
            out_features,
        )

class NetworkCNN(NetworkCNNBNN):
    def layer(self, in_features, out_features):
        return nn.Linear(
            in_features,
            out_features,
        )

class Base(nn.Module):
    def __init__(self, input_dim, architecture=[256, 128, 64], dropout=None):
        super(Base, self).__init__()

        modules = [
            nn.Linear(input_dim, architecture[0]),
            nn.ReLU(),
        ]
        for i in range(len(architecture) -1):
            if dropout:
                modules.append(nn.Dropout(p=dropout))
            modules += [
                nn.Linear(architecture[i], architecture[i+1]),
                nn.ReLU(),
            ]
        self.fc = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.fc(x)

class InverseBase(nn.Module):
    def __init__(self, output_dim, architecture=[64, 128, 256], dropout=None):
        super(InverseBase, self).__init__()

        modules = []
        for i in range(len(architecture) - 1):
            if dropout:
                modules.append(nn.Dropout(p=dropout))
            modules += [
                nn.Linear(architecture[i], architecture[i+1]),
                nn.ReLU(),
            ]
        modules += [
            nn.Linear(architecture[-1], output_dim),
            nn.ReLU(),
        ]
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

class NetworkAE(nn.Module):
    def __init__(self,
        state_stack: int,
        obs_dim: int,
        nb_actions: int,
        obs_encoder_arc: "list[int]"=[64, 32],
        act_encoder_arc: "list[int]"=[4, 16],
        shared_encoder_arc: "list[int]"=[512, 512],
        obs_decoder_arc: "list[int]"=[32, 64],
        act_decoder_arc: "list[int]"=[16, 4],
        shared_decoder_arc: "list[int]"=[512, 512],
        covar_decoder_arc: "list[int]"=[512, 1024, 2048],
        latent_dim: int=8,
        act_loss_weight: float = 1,
        obs_loss_weight: float = 1,
        prob_loss_weight: float = 0.1,
    ):
        super(NetworkAE, self).__init__()

        self.latent_dim = latent_dim
        self.state_stack = state_stack
        self.obs_dim = obs_dim
        self.nb_actions = nb_actions
        self.obs_encoder_arc = obs_encoder_arc
        self.act_encoder_arc = act_encoder_arc
        self.obs_decoder_arc = obs_decoder_arc
        self.act_decoder_arc = act_decoder_arc

        self.prob_loss_weight = prob_loss_weight
        self.act_loss_weight = act_loss_weight
        self.obs_loss_weight = obs_loss_weight
        self.obs_loss = nn.MSELoss()
        self.act_loss = nn.CrossEntropyLoss()
        self.loss = nn.GaussianNLLLoss()

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Flatten(),
            Base(state_stack*obs_dim, architecture=obs_encoder_arc)
        )
        self.act_encoder = nn.Sequential(
            nn.Flatten(),
            Base(1, architecture=act_encoder_arc)
        )
        self.shared_encoder = nn.Sequential(
            Base(act_encoder_arc[-1] + obs_encoder_arc[-1], architecture=shared_encoder_arc)
        )
        self.encoding = Base(shared_encoder_arc[-1], architecture=[latent_dim])

        # Decoders
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, shared_decoder_arc[0]),
            nn.ReLU(),
            InverseBase(obs_decoder_arc[0] + act_decoder_arc[0], architecture=shared_decoder_arc),
        )
        self.obs_decoder = InverseBase(obs_decoder_arc[-1], architecture=obs_decoder_arc[:-1])
        self.act_decoder = InverseBase(act_decoder_arc[-1], architecture=act_decoder_arc[:-1])
        
        # Decoding distribution parameters
        self.obs_mu = nn.Linear(obs_decoder_arc[-1], state_stack * obs_dim)
        covar_decoder_arc.insert(0, obs_decoder_arc[0] + act_decoder_arc[0])
        self.covar_dim = state_stack * obs_dim + nb_actions
        self.covar = nn.Sequential(
            InverseBase(covar_decoder_arc[-1], architecture=covar_decoder_arc[:-1]),
            nn.Linear(covar_decoder_arc[-1], self.covar_dim),
            nn.Softplus(),
        )

        self.act_mu = nn.Linear(act_decoder_arc[-1], nb_actions)

    def encode(self, obs, act):
        x = self.obs_encoder(obs)
        y = self.act_encoder(act)
        z = torch.cat((x, y), dim=-1)
        z = self.shared_encoder(z)
        return self.encoding(z)

    def decode(self, x):
        x = self.shared_decoder(x)
        obs = self.obs_decoder(x[:, :self.obs_decoder_arc[0]])
        act = self.act_decoder(x[:, -self.act_decoder_arc[0]:])

        obs_mu = self.obs_mu(obs)
        act_mu = self.act_mu(act)

        covar = torch.diag_embed(self.covar(x) + 0.5)
        return obs_mu, act_mu, covar

    def forward(self, obs, act):
        z = self.encode(obs, act)
        reconst_obs, reconst_act, covar = self.decode(z)
        return [reconst_obs, reconst_act, covar, (obs, act)]

    def log_prob(self, obs, act):
        obs_mu, act_mu, covar = self(obs, act)[:3]
        return -self.log_prob_loss(obs_mu, obs, act_mu, act, covar)

    def log_prob_loss(self, obs_mu, obs, act_mu, act, covar):
        one_hot_act = nn.functional.one_hot(act.squeeze(dim=1).long(), num_classes=self.nb_actions)
        target_ = torch.cat((torch.flatten(obs, start_dim=1), one_hot_act), dim=-1)
        mu = torch.cat((obs_mu, act_mu), dim=-1)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covar)
        log_prob = distribution.log_prob(target_ / 100000).sum()
        return -log_prob

    def loss_function(self, *args, **kwargs) -> dict:
        obs_mu = args[0]
        act_mu = args[1]
        covar = args[2]
        obs, act = args[3]

        obs_loss = self.obs_loss(obs_mu, torch.flatten(obs, start_dim=1))
        act_loss = self.act_loss(act_mu, act.squeeze(dim=1).long())
        prob_loss = self.log_prob_loss(obs_mu, obs, act_mu, act, covar)

        loss = self.act_loss_weight * act_loss + self.obs_loss_weight * obs_loss + prob_loss * self.prob_loss_weight

        l = {
            'loss': loss,
            'Obs Loss': obs_loss.detach(),
            'Act Loss': act_loss.detach(),
            'Prob Loss': prob_loss.detach(),
        }
        return l




if __name__ == "__main__":
    batch_size_ = 16
    window_length_ = 10
    nb_ego_states_ = 5
    nb_states_per_vehicle_ = 4
    nb_vehicles_ = 5
    nb_actions_ = 10
    nb_convolutional_layers_ = 3
    nb_conv_filters_ = 3
    nb_hidden_fc_layers_ = 3
    nb_hidden_neurons_ = 50
    duel_ = True
    prior_ = True
    prior_scale_factor_ = 100
    duel_type_ = "max"
    activation_ = "relu"
    nb_inputs_ = nb_ego_states_ + nb_states_per_vehicle_ * nb_vehicles_
    net_input = torch.rand((batch_size_, window_length_, nb_inputs_))

    net = NetworkCNNBNN(
        nb_ego_states_,
        nb_states_per_vehicle_,
        nb_vehicles_,
        nb_actions_,
        nb_convolutional_layers_,
        nb_conv_filters_,
        nb_hidden_fc_layers_,
        nb_hidden_neurons_,
        duel_,
        prior_,
        prior_scale_factor_,
        duel_type_,
        activation_,
        window_length_,
    )

    nb_outputs_ = 10
    nb_hidden_layers_ = 5
    net2 = NetworkMLPBNN(
        nb_inputs_,
        nb_outputs_,
        nb_hidden_layers_,
        nb_hidden_neurons_,
        duel_,
        prior_,
        prior_scale_factor_,
        duel_type_,
        activation_,
        window_length_,
    )

    print(net2(net_input).shape)
