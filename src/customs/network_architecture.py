import torch
import torch.nn as nn
import torchbnn as bnn

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
        raise Exception("Not Implemented yet")

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
        raise Exception("Not Implemented yet")

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

        if len(architecture) == 0:
            self.fc = lambda x: x
        elif len(architecture) == 1:
            self.fc = nn.Linear(input_dim, architecture[0])
        else:
            modules = [
                nn.Linear(input_dim, architecture[0]),
                nn.ReLU(),
            ]
            for i in range(len(architecture) - 2):
                if dropout:
                    modules.append(nn.Dropout(p=dropout))
                modules += [
                    nn.Linear(architecture[i], architecture[i + 1]),
                    nn.ReLU(),
                ]
            modules += [
                nn.Linear(architecture[-2], architecture[-1]),
            ]
            self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class InverseBase(nn.Module):
    def __init__(self, output_dim, architecture=[64, 128, 256], dropout=None):
        super(InverseBase, self).__init__()

        if len(architecture) == 0:
            self.fc = lambda x: x
        elif len(architecture) == 1:
            self.fc = nn.Linear(architecture[0], output_dim)
        else:
            modules = []
            for i in range(len(architecture) - 1):
                if dropout:
                    modules.append(nn.Dropout(p=dropout))
                modules += [
                    nn.Linear(architecture[i], architecture[i + 1]),
                    nn.ReLU(),
                ]
            modules += [
                nn.Linear(architecture[-1], output_dim),
            ]
            self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)

class Cholesky(nn.Module):
    def __init__(self, input_shape, output_shape=6, min_value=1e-8):
        super(Cholesky, self).__init__()
        self.inds_a, self.inds_b = torch.tril_indices(output_shape, output_shape)
        self.is_diag = self.inds_a == self.inds_b
        self.output_shape = output_shape
        self.positive_fun = torch.nn.Softplus()
        self.min_value = torch.tensor(min_value)
        self.register_buffer('_min_value', self.min_value)
        mid_shape = int(output_shape * (output_shape + 1) / 2)
        self.in_layer = nn.Linear(input_shape, mid_shape)

    def forward(self, x):
        x = self.in_layer(x)
        x = torch.where(self.is_diag, self.positive_fun(x) + self.min_value, x)
        L = torch.zeros((x.shape[0], self.output_shape, self.output_shape),
                        dtype=x.dtype)
        L[:, self.inds_a, self.inds_b] = x
        LT = L.transpose(1, 2)
        out = L @ LT
        return out

class NetworkAE(nn.Module):
    def __init__(
        self,
        state_stack: int,
        obs_dim: int,
        actions: int,
        obs_encoder_arc: "list[int]" = [64, 32],
        act_encoder_arc: "list[int]" = [4, 16],
        shared_encoder_arc: "list[int]" = [512, 512],
        obs_decoder_arc: "list[int]" = [32, 64],
        act_decoder_arc: "list[int]" = [16, 4],
        shared_decoder_arc: "list[int]" = [512, 512],
        covar_decoder_arc: "list[int]" = [512, 1024, 2048],
        latent_dim: int = 8,
        act_loss_weight: float = 1,
        obs_loss_weight: float = 1,
        prob_loss_weight: float = 0.1,
    ):
        super(NetworkAE, self).__init__()

        self.latent_dim = latent_dim
        self.state_stack = state_stack
        self.obs_dim = obs_dim
        self.actions = torch.Tensor(actions)
        self.act_dim = len(actions[0])
        self.nb_actions = len(actions)
        self.obs_encoder_arc = obs_encoder_arc
        self.act_encoder_arc = act_encoder_arc
        self.obs_decoder_arc = obs_decoder_arc
        self.act_decoder_arc = act_decoder_arc

        self.prob_loss_weight = prob_loss_weight
        self.act_loss_weight = act_loss_weight
        self.obs_loss_weight = obs_loss_weight
        self.obs_loss = nn.MSELoss()
        self.act_loss = nn.MSELoss()
        self.loss = nn.GaussianNLLLoss()

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Flatten(), Base(state_stack * obs_dim, architecture=obs_encoder_arc), nn.ReLU(),
        )
        self.act_encoder = nn.Sequential(
            nn.Flatten(), Base(self.act_dim, architecture=act_encoder_arc), nn.ReLU(),
        )
        self.shared_encoder = nn.Sequential(
            Base(
                act_encoder_arc[-1] + obs_encoder_arc[-1],
                architecture=shared_encoder_arc,
            ),
            nn.Linear(shared_encoder_arc[-1], latent_dim),
        )

        # Decoders
        shared_decoder_out = obs_decoder_arc[0] + act_decoder_arc[0]
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, shared_decoder_arc[0]),
            nn.ReLU(),
            InverseBase(
                shared_decoder_out, architecture=shared_decoder_arc
            ),
            nn.ReLU(),
        )
        self.obs_mu = nn.Sequential(
            InverseBase(
                obs_decoder_arc[-1], architecture=obs_decoder_arc[:-1]
            ),
            nn.ReLU(),
            nn.Linear(obs_decoder_arc[-1], state_stack * obs_dim),
        )
        self.act_mu = nn.Sequential(
            InverseBase(
                act_decoder_arc[-1], architecture=act_decoder_arc[:-1]
            ),
            nn.ReLU(),
            nn.Linear(act_decoder_arc[-1], self.act_dim),
        )

        # Decoding distribution parameters
        self.covar_dim = state_stack * obs_dim + self.act_dim
        self.covar = nn.Sequential(
            InverseBase(covar_decoder_arc[-1], architecture=[shared_decoder_out] + covar_decoder_arc[:-1]),
            nn.ReLU(),
            Cholesky(covar_decoder_arc[-1], self.covar_dim),
        )

    def encode(self, obs: torch.Tensor, act: torch.Tensor):
        x = self.obs_encoder(obs)
        y = self.act_encoder(act)
        z = torch.cat((x, y), dim=-1)
        return self.shared_encoder(z)

    def decode(self, x: torch.Tensor):
        x = self.shared_decoder(x)
        obs_mu = self.obs_mu(x[:, : self.obs_decoder_arc[0]])
        act_mu = self.act_mu(x[:, -self.act_decoder_arc[0] :])

        covar = self.covar(x)
        return obs_mu, act_mu, covar

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        act_ = (
            torch.index_select(self.actions.to(act.device), 0, act.squeeze(dim=1).long())
            .float()
        )
        z = self.encode(obs, act_)
        reconst_obs, reconst_act, covar = self.decode(z)
        return [reconst_obs, reconst_act, covar, (obs, act)]

    def nll_loss(
        self,
        obs_mu: torch.Tensor,
        obs: torch.Tensor,
        act_mu: torch.Tensor,
        act: torch.Tensor,
        covar: torch.Tensor,
        **kwargs
    ):
        act_ = (
            torch.index_select(self.actions.to(act.device), 0, act.squeeze(dim=1).long())
            .float()
            .to(act.device)
        )
        target_ = torch.cat((torch.flatten(obs, start_dim=1), act_), dim=-1)
        mu = torch.cat((obs_mu, act_mu), dim=-1)
        if True:
            for i in range(obs_mu.shape[0]):
                mu_i, covar_i, target_i = mu[i], covar[i], target_[i]
                distribution = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        mu_i, covar_i
                    )
                )
                if i == 0:
                    log_prob = distribution.log_prob(target_i)
                else:
                    log_prob += distribution.log_prob(target_i)
        else:
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mu, covar
            )
            log_prob = distribution.log_prob(target_)
        return -log_prob

    def loss_function(self, *args, **kwargs) -> dict:
        obs_mu = args[0]
        act_mu = args[1]
        covar = args[2]
        obs, act = args[3]

        act_ = (
            torch.index_select(self.actions.to(act.device), 0, act.squeeze(dim=1).long())
            .float()
            .to(act.device)
        )

        obs_loss = self.obs_loss(obs_mu, torch.flatten(obs, start_dim=1))
        act_loss = self.act_loss(act_mu, act_)
        prob_loss = self.nll_loss(obs_mu, obs, act_mu, act, covar)

        loss = (
            self.act_loss_weight * act_loss
            + self.obs_loss_weight * obs_loss
            + prob_loss * self.prob_loss_weight
        )

        l = {
            "loss": loss,
            "Obs Loss": obs_loss.detach(),
            "Act Loss": act_loss.detach(),
            "Prob Loss": prob_loss.detach(),
        }
        return l
    

class NetworkAESimple(nn.Module):
    def __init__(
        self,
        stack: int,
        input_dim: int,
        encoder_arc: "list[int]" = [64, 32],
        shared_decoder_arc: "list[int]" = [16.32],
        decoder_arc: "list[int]" = [32, 64],
        covar_decoder_arc: "list[int]" = [512, 1024, 2048],
        latent_dim: int = 8,
        input_loss_weight: float = 1,
        prob_loss_weight: float = 0.1,
    ):
        super(NetworkAESimple, self).__init__()

        self.latent_dim = latent_dim
        self.stack = stack
        self.input_dim = input_dim
        self.encoder_arc = encoder_arc
        self.decoder_arc = decoder_arc
        self.shared_decoder_arc = shared_decoder_arc

        self.prob_loss_weight = prob_loss_weight
        self.input_loss_weight = input_loss_weight
        self.input_loss = nn.MSELoss()
        self.loss = nn.GaussianNLLLoss()

        self.input_dim = stack * input_dim

        # Encoders
        self.encoder = nn.Sequential(
            nn.Flatten(),
            Base(self.input_dim, architecture=encoder_arc + [latent_dim])
        )

        # Decoders
        self.shared_decoder = nn.Sequential(
            InverseBase(
                shared_decoder_arc[-1], architecture=[latent_dim] + shared_decoder_arc[:-1]
            ),
            nn.ReLU()
        )
        self.mu = InverseBase(
            self.input_dim, architecture=[shared_decoder_arc[-1]] + decoder_arc,
        )
        covar_decoder_arc.insert(0, shared_decoder_arc[-1])
        self.covar = nn.Sequential(
            InverseBase(covar_decoder_arc[-1], architecture=covar_decoder_arc[:-1]),
            nn.ReLU(),
            Cholesky(covar_decoder_arc[-1], self.input_dim),
        )

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, x: torch.Tensor):
        x = self.shared_decoder(x)
        mu = self.mu(x)
        covar = self.covar(x)
        return mu, covar

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        reconst_x, covar = self.decode(z)
        return [reconst_x, covar, x]

    def nll_loss(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
        covar: torch.Tensor,
        **kwargs
    ):
        if True:
            for i in range(x.shape[0]):
                mu_i, covar_i, x_i = mu[i], covar[i], x[i]
                distribution = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        mu_i, covar_i
                    )
                )
                if i == 0:
                    log_prob = distribution.log_prob(x_i)
                else:
                    log_prob += distribution.log_prob(x_i)
        else:
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mu, covar
            )
            log_prob = distribution.log_prob(x)
        return -log_prob

    def loss_function(self, *args, **kwargs) -> dict:
        mu = args[0]
        covar = args[2]
        x = args[3]

        input_loss = self.input_loss(mu, torch.flatten(x, start_dim=1))
        prob_loss = self.nll_loss(mu, x, covar)

        loss = (
            input_loss * self.input_loss_weight
            + prob_loss * self.prob_loss_weight
        )

        l = {
            "loss": loss,
            "Input Loss": input_loss.detach(),
            "Prob Loss": prob_loss.detach(),
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
