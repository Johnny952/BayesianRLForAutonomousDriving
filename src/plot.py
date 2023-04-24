import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.ndimage import gaussian_filter1d

PERCENTILE_DOWN = 1
PERCENTILE_UP = 99
EPSILON = 1e-15


def read_test(path):
    thresholds = []
    rewards = []
    collision_rates = []
    nb_safe_actions = []
    nb_safe_action_hard = []
    collision_speeds = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            thresholds.append(float(row[0]))
            rewards.append(float(row[1]))
            collision_rates.append(float(row[2]))
            nb_safe_actions.append(float(row[3]))
            nb_safe_action_hard.append(float(row[4]))
            collision_speeds.append(float(row[5]))
    return (
        np.array(thresholds),
        np.array(rewards),
        np.array(collision_rates),
        np.array(nb_safe_actions),
        np.array(nb_safe_action_hard),
        np.array(collision_speeds),
    )


def read_file(path, unc_normalized=True, negative_unc=False):
    steps = []
    uncertainty = []
    collision_rate = []
    collision_speed = []
    rewards = []

    max_unc = -1e10
    min_unc = 1e10
    mean_max = -1e10
    mean_min = 1e10
    with h5py.File(path, "r") as file:
        for step_key, step in file.items():
            unc, nb_steps = step["uncertainties"][()], step["steps"][()]
            total_steps = np.sum(nb_steps)
            unc = unc / total_steps
            if negative_unc:
                unc = -unc
            max_ = np.max(unc)
            min_ = np.min(unc)
            mean_ = np.mean(unc)
            if max_ > max_unc:
                max_unc = max_
            if min_ < min_unc:
                min_unc = min_
            if mean_ > mean_max:
                mean_max = mean_
            if mean_ < mean_min:
                mean_min = mean_

    uncertainty_mean = []
    uncertainty_up = []
    uncertainty_low = []
    uncertainty_std = []
    with h5py.File(path, "r") as file:
        for step_key, step in file.items():
            steps.append(int(step_key))

            unc, rew, col, col_speed, nb_steps = (
                step["uncertainties"][()],
                step["reward"][()],
                step["collision"][()],
                step["collision_speed"][()],
                step["steps"][()],
            )
            total_steps = np.sum(nb_steps)
            unc = unc / total_steps
            if negative_unc:
                unc = -unc
            if unc_normalized and max_unc != min_unc:
                # unc = (unc - min_unc) / (max_unc - min_unc + EPSILON)
                unc = (unc - mean_min) / (mean_max - mean_min + EPSILON)

            uncertainty.append(unc)
            uncertainty_mean.append(np.mean(unc))
            uncertainty_up.append(np.percentile(unc, PERCENTILE_UP))
            uncertainty_low.append(np.percentile(unc, PERCENTILE_DOWN))
            uncertainty_std.append(np.std(unc))

            # rew / np.sum(nb_steps)
            norm_r = rew / nb_steps
            rewards.append((np.mean(norm_r) + 10) / 11)

            rate = np.sum(col) / len(col)
            collision_rate.append(rate)

            coll_s = col_speed[col_speed > 0]
            if len(coll_s) == 0:
                collision_speed.append(0)
            else:
                collision_speed.append(np.mean(coll_s))

    steps = np.array(steps)
    uncertainty_mean = np.array(uncertainty_mean, dtype=np.float16)
    uncertainty_up = np.array(uncertainty_up, dtype=np.float16)
    uncertainty_low = np.array(uncertainty_low, dtype=np.float16)
    uncertainty_std = np.array(uncertainty_std, dtype=np.float16)

    collision_rate = np.array(collision_rate, dtype=np.float16)
    collision_speed = np.array(collision_speed, dtype=np.float16)
    rewards = np.array(rewards, dtype=np.float16)

    arr1inds = steps.argsort()
    steps = steps[arr1inds]
    uncertainty_mean = uncertainty_mean[arr1inds]
    uncertainty_up = uncertainty_up[arr1inds]
    uncertainty_low = uncertainty_low[arr1inds]
    uncertainty_std = uncertainty_std[arr1inds]
    collision_rate = collision_rate[arr1inds]
    collision_speed = collision_speed[arr1inds]
    rewards = rewards[arr1inds]

    return (
        steps,
        (uncertainty_mean, uncertainty_up, uncertainty_low, uncertainty_std),
        (collision_rate, collision_speed),
        rewards,
    )


def plot_train(model_nb=-1):
    plt.figure(figsize=(13, 8))

    # Rewards
    ax2 = plt.subplot(222)
    for model in models:
        model_name = model["name"]
        rewards = []
        steps = []
        for path in model["paths"]:
            step, _, _, reward = read_file(path)
            rewards.append(reward)
            steps.append(step)
        rewards = np.array(rewards)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        steps = steps[0]

        plt.plot(
            steps,
            rewards_mean,
            color=model["color"],
            label=f"Mean {model_name}",
        )
        plt.fill_between(
            steps,
            (rewards_mean - rewards_std),
            (rewards_mean + rewards_std),
            color=model["color"],
            alpha=0.2,
            # label="Std",
        )
    ax2.spines["top"].set_visible(False)
    plt.xlabel("Traning step", fontsize=14)
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.legend()

    # Uncertainty
    ax1 = plt.subplot(221, sharex=ax2)
    for model in models:
        model_name = model["name"]
        if model["show_uncertainty"]:
            (
                steps,
                (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std),
                _,
                _,
            ) = read_file(
                model["paths"][model_nb], negative_unc=model["negative_uncertainty"]
            )
            plt.plot(
                steps,
                uncertainty,
                color=model["color"],
                label=f"Mean {model_name}",
            )
            # plt.fill_between(
            #     steps,
            #     (uncertainty - uncertainty_std),
            #     (uncertainty + uncertainty_std),
            #     color=model["color"],
            #     alpha=0.2,
            #     label="Std",
            # )
            # plt.fill_between(
            #     steps,
            #     (uncertainty_up),
            #     (uncertainty_low),
            #     color=model["color"],
            #     alpha=0.1,
            # )
    ax1.spines["top"].set_visible(False)
    plt.xlabel("Traning step", fontsize=14)
    plt.ylabel("Normalized Uncertainty", fontsize=14)
    plt.ylim((0, 0.1))
    # plt.yscale('log')
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)

    # Collision rate
    ax3 = plt.subplot(223, sharex=ax2)
    for model in models:
        model_name = model["name"]
        steps, _, (collision_rate, _), _ = read_file(model["paths"][model_nb])
        plt.plot(
            steps,
            1 - collision_rate,
            color=model["color"],
            label=f"Mean {model_name}",
        )
    plt.xlabel("Traning step", fontsize=14)
    plt.ylabel("Collision Free Episodes", fontsize=14)
    ax3.spines["top"].set_visible(False)
    plt.ylim((-0.05, 1.05))
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # Collision Speed
    ax4 = plt.subplot(224, sharex=ax2)
    for model in models:
        model_name = model["name"]
        steps, _, (_, collision_speed), _ = read_file(model["paths"][model_nb])
        plt.plot(
            steps,
            collision_speed,
            color=model["color"],
            label=f"Mean {model_name}",
            alpha=0.1,
        )
        filtered_speeds = gaussian_filter1d(collision_speed.astype(np.float32), sigma=2)
        plt.plot(
            steps,
            filtered_speeds,
            color=model["color"],
            label=f"Mean {model_name}",
            alpha=1,
        )
        # plt.fill_between(
        #     steps,
        #     (collision_speed_up),
        #     (collision_speed_low),
        #     color=model["color"],
        #     alpha=0.1,
        # )
    plt.xlabel("Traning step", fontsize=14)
    plt.ylabel("Collision Speed", fontsize=14)
    ax4.spines["top"].set_visible(False)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)

    # plt.show()
    plt.savefig("./videos/fig.png")
    plt.close()


def plot_test(scenario="rerun_test_scenarios"):
    # Rewards vs collision rate
    fig = plt.figure()
    fig, axs = plt.subplots(ncols=3, nrows=2)
    grids = axs[0, 0].get_gridspec()
    for axis in axs[:2, :2].flat:
        axis.remove()
    # fig, ax = plt.subplots(1, 3)
    fig.set_figheight(12)
    fig.set_figwidth(18)

    # ax2 = plt.subplot(2, 3, 3)
    # ax3 = plt.subplot(2, 3, 6)
    # ax1 = plt.subplot(2, 1, (1, 2, 4, 5))
    ax1 = fig.add_subplot(grids[:2, :2])
    # ax4 = ax1.twinx()
    ax2 = axs[0, 2]
    ax3 = axs[1, 2]

    plot_models = [0, 3]

    model = models[plot_models[0]]
    (
        _,
        rewards,
        collision_rates,
        nb_safe_actions,
        nb_safe_action_hard,
        collision_speeds,
    ) = read_test(model["multiple_test"][scenario])
    idc = np.argsort(collision_rates)
    sorted_rates, sorted_rewards = collision_rates[idc], rewards[idc]
    # filtered_rewards = gaussian_filter1d(sorted_rewards.astype(np.float32), sigma=0.9)
    # ax1.plot(sorted_rates, filtered_rewards, color=model["color"], label=model["name"], alpha=1)
    unique_rates, unique_idxs = np.unique(sorted_rates, return_index=True)
    unique_rewards = sorted_rewards[unique_idxs]
    ax1.plot(
        unique_rates, unique_rewards, color=model["color"], label=model["name"], alpha=1
    )
    ax2.plot(rewards, color=model["color"], label=model["name"], alpha=1)
    ax3.plot(collision_rates, color=model["color"], label=model["name"], alpha=1)

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    # ax1.tick_params(axis='y', colors=model["color"])
    ax1.set_ylabel("Rewards", fontsize=14)
    # ax1.set_ylabel(f'Rewards {model["name"]}', fontsize=14, color=model["color"])

    model = models[plot_models[1]]
    (
        _,
        rewards,
        collision_rates,
        nb_safe_actions,
        nb_safe_action_hard,
        collision_speeds,
    ) = read_test(model["multiple_test"][scenario])
    idc = np.argsort(collision_rates)
    sorted_rates, sorted_rewards = collision_rates[idc], rewards[idc]
    # filtered_rewards = gaussian_filter1d(sorted_rewards.astype(np.float32), sigma=0.9)
    # ax1.plot(sorted_rates, filtered_rewards, color=model["color"], label=model["name"], alpha=1)
    unique_rates, unique_idxs = np.unique(sorted_rates, return_index=True)
    unique_rewards = sorted_rewards[unique_idxs]
    # ax4.plot(unique_rates, unique_rewards, color=model["color"], label=model["name"], alpha=1)
    ax1.plot(
        unique_rates, unique_rewards, color=model["color"], label=model["name"], alpha=1
    )
    ax2.plot(rewards, color=model["color"], label=model["name"], alpha=1)
    ax3.plot(collision_rates, color=model["color"], label=model["name"], alpha=1)

    # ax4.set_xlim(left=0)
    # ax4.set_ylim(bottom=0)
    # ax4.tick_params(axis='y', colors=model["color"])
    # ax4.set_ylabel(f'Rewards {model["name"]}', fontsize=14, color=model["color"])

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_xlabel("Collision Rate", fontsize=14)
    ax1.legend()
    ax1.grid()

    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    ax2.set_ylabel("Reward", fontsize=14)
    ax2.set_xlabel("Threshold", fontsize=14)

    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax3.set_ylim(bottom=0)
    ax3.set_xlim(left=0)
    ax3.set_ylabel("Collision Rate", fontsize=14)
    ax3.set_xlabel("Threshold", fontsize=14)

    # plt.show()
    plt.savefig("./videos/compar.png")
    plt.close()


if __name__ == "__main__":
    models = [
        {
            "paths": [
                "./logs/train_agent_20230323_235314_dqn_6M_v3/data.hdf5",
            ],
            "multiple_test": {
                "rerun_test_scenarios": None,
                "standstill": None,
                "fast_overtaking": None,
            },
            "name": "Standard DQN",
            "show_uncertainty": False,
            "negative_uncertainty": False,
            "color": "blue",
        },
        {
            "paths": [
                "./logs/train_agent_20230323_235219_rpf_6M_v3/data.hdf5",
            ],
            "multiple_test": {
                "rerun_test_scenarios": None,
                "standstill": None,
                "fast_overtaking": None,
            },
            "name": "Ensemble RPF DQN",
            "show_uncertainty": True,
            "negative_uncertainty": False,
            "color": "red",
        },
        {
            "paths": [
                "./logs/train_agent_20230405_010753_bnn_6M_v4/data.hdf5",
            ],
            "multiple_test": {
                "rerun_test_scenarios": None,
                "standstill": None,
                "fast_overtaking": None,
            },
            "name": "BNN DQN",
            "show_uncertainty": True,
            "negative_uncertainty": False,
            "color": "orange",
        },
        {
            "paths": [
                "./logs/train_agent_20230404_002949_ae_6M_v7/data.hdf5",
            ],
            "multiple_test": {
                "rerun_test_scenarios": None,#"./logs/train_agent_20230404_002949_ae_6M_v7/rerun_test_scenarios.csv",
                "standstill": None,
                "fast_overtaking": None,
            },
            "name": "AE DQN",
            "show_uncertainty": True,
            "negative_uncertainty": True,
            "color": "green",
        },
    ]

    plot_train()
    plot_test()
