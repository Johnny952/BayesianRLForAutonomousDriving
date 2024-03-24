import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.ndimage import gaussian_filter1d

PERCENTILE_DOWN = 1
PERCENTILE_UP = 99
EPSILON = 1e-15


def collapse_duplicated(*arrays, collapse_by=0, sort_by=0, reduction=np.max):
    collapse_array = arrays[collapse_by]
    unique, unique_idcs = np.unique(collapse_array, return_index=True)
    new_arrays = []
    for array in arrays:
        new_array = np.zeros(unique.shape)
        for i, unique_value in enumerate(unique):
            filter_ = collapse_array == unique_value
            new_array[i] = reduction(array[filter_])
        new_arrays.append(new_array)

    sorted_array_idcs = new_arrays[sort_by].argsort()
    sorted_arrays = []
    for array in new_arrays:
        sorted_arrays.append(array[sorted_array_idcs])
    return unique_idcs, sorted_arrays


def sort_duplicated(*arrays, sort_by=None, collapse_by=0):
    collapse_array = arrays[collapse_by]
    unique, unique_idcs = np.unique(collapse_array, return_index=True)
    new_arrays = []
    for array in arrays:
        new_array = np.array([])
        for i, unique_value in enumerate(unique):
            filter_ = collapse_array == unique_value
            new_array = np.concatenate((new_array, np.sort(array[filter_])))
        new_arrays.append(new_array)

    if sort_by is not None:
        sorted_array_idcs = sort_by.argsort()
    else:
        sorted_array_idcs = new_arrays[0].argsort()
    sorted_arrays = []
    for array in new_arrays:
        sorted_arrays.append(array[sorted_array_idcs])
    return unique_idcs, sorted_arrays


def read_test(path):
    thresholds = []
    rewards = []
    collision_rates = []
    nb_safe_actions = []
    nb_safe_action_hard = []
    collision_speeds = []
    steps = []
    stop_events = []
    fast_events = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            thresholds.append(float(row[0]))
            rewards.append(float(row[1]))
            collision_rates.append(float(row[2]))
            nb_safe_actions.append(float(row[3]))
            nb_safe_action_hard.append(float(row[4]))
            collision_speeds.append(float(row[5]))
            steps.append(float(row[6]))
            if len(row) == 9:
                stop_events.append(float(row[7]))
                fast_events.append(float(row[8]))
    return (
        np.array(thresholds),
        np.array(rewards),
        np.array(collision_rates),
        np.array(nb_safe_actions),
        np.array(nb_safe_action_hard),
        np.array(collision_speeds),
        np.array(steps),
        np.array(stop_events),
        np.array(fast_events),
    )


def read_test2(path, ep_type=int):
    thresholds = []
    episodes = []
    uncert = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            thresholds.append(float(row[0]))
            episodes.append(ep_type(row[1]))
            uncert.append([float(d) for d in row[2:]])
    return thresholds, episodes, uncert


def read_file(path, unc_normalized=True):
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

    # mean_min = 0

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


def plot_train():
    # plt.figure(figsize=(13, 8))

    # Rewards
    plt.figure(figsize=(9.6, 6.4)) #3 2
    ax2 = plt.subplot(111)
    for model in models:
        if model["train"]["show"]:
            model_name = model["name"]
            model_color = model["color"]
            paths = model["train"]["paths"]
            rewards = []
            steps = []
            for path in paths:
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
                color=model_color,
                label=f"Mean {model_name}",
            )
            plt.fill_between(
                steps,
                (rewards_mean - rewards_std),
                (rewards_mean + rewards_std),
                color=model_color,
                alpha=0.2,
                # label="Std",
            )
    ax2.spines["top"].set_visible(False)
    plt.xlabel("Traning step", fontsize=24)
    plt.ylabel("Normalized Reward", fontsize=24)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.savefig("./videos/train_rewards.png")
    plt.close()

    # Uncertainty
    plt.figure(figsize=(9.6, 6.4))
    ax1 = plt.subplot(111)
    for model in models:
        if model["train"]["show"]:
            model_name = model["name"]
            model_color = model["color"]
            paths = model["train"]["paths"]
            model_nb = model["train"]["model_uncertainty"]
            if model["train"]["show_uncertainty"]:
                (
                    steps,
                    (uncertainty, _, _, uncertainty_std),
                    _,
                    _,
                ) = read_file(paths[model_nb])
                plt.plot(
                    steps,
                    uncertainty,
                    color=model_color,
                    label=f"Mean {model_name}",
                )
                plt.fill_between(
                    steps,
                    (uncertainty - uncertainty_std),
                    (uncertainty + uncertainty_std),
                    color=model_color,
                    alpha=0.2,
                    #label="Std",
                )
                # plt.fill_between(
                #     steps,
                #     (uncertainty_up),
                #     (uncertainty_low),
                #     color=model_color,
                #     alpha=0.1,
                # )
    ax1.spines["top"].set_visible(False)
    plt.xlabel("Traning step", fontsize=24)
    plt.ylabel("Normalized Uncertainty", fontsize=24)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylim((0, 1))
    plt.legend(fontsize=16)
    # plt.yscale('log')
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.savefig("./videos/train_uncertainties.png")
    plt.close()

    # Collision rate
    plt.figure(figsize=(9.6, 6.4))
    ax3 = plt.subplot(111)
    for model in models:
        if model["train"]["show"]:
            model_name = model["name"]
            model_color = model["color"]
            paths = model["train"]["paths"]
            model_nb = model["train"]["model_uncertainty"]
            steps, _, (collision_rate, _), _ = read_file(paths[model_nb])
            plt.plot(
                steps,
                1 - collision_rate,
                color=model_color,
                label=f"Mean {model_name}",
            )
    plt.xlabel("Traning step", fontsize=24)
    plt.ylabel("Collision Free Episodes", fontsize=24)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    ax3.spines["top"].set_visible(False)
    plt.ylim((-0.05, 1.05))
    plt.legend(fontsize=16)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.savefig("./videos/train_colrate.png")
    plt.close()

    # Collision Speed
    plt.figure(figsize=(9.6, 6.4))
    ax4 = plt.subplot(111)
    for model in models:
        if model["train"]["show"]:
            model_name = model["name"]
            model_color = model["color"]
            paths = model["train"]["paths"]
            model_nb = model["train"]["model_uncertainty"]
            steps, _, (_, collision_speed), _ = read_file(paths[model_nb])
            plt.plot(
                steps,
                collision_speed,
                color=model_color,
                #label=f"Mean {model_name}",
                alpha=0.1,
            )
            filtered_speeds = gaussian_filter1d(collision_speed.astype(np.float32), sigma=2)
            plt.plot(
                steps,
                filtered_speeds,
                color=model_color,
                label=f"Mean {model_name}",
                alpha=1,
            )
            # plt.fill_between(
            #     steps,
            #     (collision_speed_up),
            #     (collision_speed_low),
            #     color=model_color,
            #     alpha=0.1,
            # )
    plt.xlabel("Traning step", fontsize=24)
    plt.ylabel("Collision Speed", fontsize=24)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    ax4.spines["top"].set_visible(False)
    plt.legend(fontsize=16)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    plt.savefig("./videos/train_colspeed.png")
    plt.close()

def plot_rerun_test_v3():
    scenario = "rerun_test_scenarios"
    fig1, ax1 = plt.subplots(ncols=1, nrows=1)
    fig1.set_figwidth(7)
    fig1.set_figheight(7)

    fig2, ax2 = plt.subplots(ncols=1, nrows=1)
    fig2.set_figwidth(7)
    fig2.set_figheight(7)

    fig3, ax3 = plt.subplots(ncols=1, nrows=1)
    fig3.set_figwidth(7)
    fig3.set_figheight(7)

    fig4, ax4 = plt.subplots(ncols=1, nrows=1)
    fig4.set_figwidth(7)
    fig4.set_figheight(7)

    for model in models:
        base_path = model["base_path"]
        model_name = model["name"]
        model_color = model["color"]
        mark = model["ROC"]["mark"]
        scenario_path = model["ROC"]["path"]
        if model["ROC"]["use_uncertainty"]:
            (
                thresholds,
                rewards,
                collision_rates,
                nb_safe_actions,
                _,
                collision_speeds,
                _,
                _,
                _,
            ) = read_test(f"{base_path}{scenario_path}")
            print(model_name)
            print("Threshold\tReward\tSafe Actions\tCollision Rates")
            for i, thresh in enumerate(thresholds):
                print(
                    "{:.3f}\t\t{:.3f}\t{:.2f}\t\t\t{:.2f}%".format(
                        thresh,
                        rewards[i],
                        nb_safe_actions[i],
                        collision_rates[i] * 100,
                    )
                )
            print("\n\n")
            # _, [filtered_rates_s, filtered_rewards_s] = collapse_duplicated(collision_rates, rewards)
            _, [filtered_rates_s, filtered_rewards_s] = sort_duplicated(
                collision_rates, rewards
            )
            # filtered_rates_s, filtered_rewards_s = collision_rates, rewards
            _, [
                filtered_safe_action,
                filtered_rates,
                filtered_rewards,
                filtered_speeds,
            ] = collapse_duplicated(
                nb_safe_actions,
                collision_rates,
                rewards,
                collision_speeds,
                reduction=np.mean,
            )
            ax1.plot(
                filtered_rates_s,
                filtered_rewards_s,
                mark,
                color=model_color,
                label=f"{model_name}",
                alpha=1,
                linewidth=2,
            )

            ax2.plot(
                filtered_safe_action,
                filtered_rewards,
                mark,
                color=model_color,
                label=f"{model_name}",
                alpha=1,
            )
            ax3.plot(
                filtered_safe_action,
                filtered_rates,
                mark,
                color=model_color,
                label=f"{model_name}",
                alpha=1,
            )
            ax4.plot(
                filtered_safe_action,
                filtered_speeds,
                mark,
                color=model_color,
                label=f"{model_name}",
                alpha=1,
            )

        else:
            (
                _,
                rewards,
                collision_rates,
                _,
                _,
                collision_speeds,
                _,
                _,
                _,
            ) = read_test(f"{base_path}{scenario_path}")
            ax1.plot(
                collision_rates,
                rewards,
                mark,
                color=model_color,
                label=f"{model_name}",
                alpha=1,
                markersize=16,
            )
            ax2.axhline(
                y=rewards[0],
                xmin=0.0,
                label=f"{model_name}",
                color=model_color,
                linestyle="--",
            )
            ax3.axhline(
                y=collision_rates[0],
                xmin=0.0,
                label=f"{model_name}",
                color=model_color,
                linestyle="--",
            )
            ax4.axhline(
                y=collision_speeds[0],
                xmin=0.0,
                label=f"{model_name}",
                color=model_color,
                linestyle="--",
            )

    ax1.set_title(f"{scenario}", fontsize=25)
    ax1.set_xlim(left=0)
    # ax1.set_ylim(bottom=-4)
    ax1.set_ylabel("Rewards", fontsize=16)

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_xlabel("Collision Rate", fontsize=16)
    ax1.legend()
    ax1.grid()

    ax2.set_title(f"{scenario}", fontsize=25)
    # ax2.set_ylim(bottom=-4)
    ax2.set_xlim(left=0)
    ax2.legend()
    ax2.set_ylabel("Reward", fontsize=16)
    ax2.set_xlabel("Safe Action Rate", fontsize=16)

    ax3.set_title(f"{scenario}", fontsize=25)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax3.set_ylim(bottom=0)
    ax3.legend()
    ax3.set_xlim(left=0)
    ax3.set_ylabel("Collision Rate", fontsize=16)
    ax3.set_xlabel("Safe Action Rate", fontsize=16)

    ax4.set_ylim(bottom=0)
    ax4.set_xlim(left=0)
    ax4.legend()
    ax4.set_ylabel("Collision Speeds", fontsize=16)
    ax4.set_xlabel("Safe Action Rate", fontsize=16)

    # plt.show()
    fig1.savefig(f"./videos/{scenario}_compar_v3.png")

    fig2.savefig(f"./videos/{scenario}_rewards_v3.png")

    fig3.savefig(f"./videos/{scenario}_collisions_v3.png")

    fig4.savefig(f"./videos/{scenario}_speeds_v3.png")
    plt.close()


def plot_tests_v3():
    import seaborn as sn
    import pandas as pd
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    for model in models:
        if model["unc_heatmap"]["show"]:
            base_path = model["base_path"]
            model_name = model["name"]

            # fig.suptitle(model_name, fontsize=24)

            for idx, scenario in enumerate(
                ["standstill", "fast_overtaking"]
            ):  # standstill, "fast_overtaking"
                fig, ax = plt.subplots(ncols=1, nrows=1)
                fig.set_figwidth(16)
                fig.set_figheight(8)
                scenario_path = model["unc_heatmap"][scenario]
                path = f"{base_path}{scenario_path}"
                _, pos_vel, unc = read_test2(path, ep_type=float)
                max_len = len(max(unc, key=len))
                min_unc = min(unc)
                flat_unc = []
                for u in unc:
                    flat_unc += u
                print(model_name, np.min(flat_unc), np.max(flat_unc))
                unc_range = model["unc_heatmap"]["unc_range"]
                padded_unc = np.ones((len(unc), max_len)) * min_unc
                for i, row in enumerate(unc):
                    padded_unc[i, : len(row)] = row
                df_cm = pd.DataFrame(
                    padded_unc[:, ::-1],
                    columns=[max_len - i for i in range(max_len)],
                    index=[
                        int(p) if scenario == "standstill" else f"{p:.1f}"
                        for p in pos_vel
                    ],
                )
                ax_idx = 0 if scenario == "standstill" else 1
                sn.heatmap(
                    df_cm.T,
                    annot=False,
                    ax=ax,
                    cbar_kws={"label": "Uncertainty"},
                    vmin=unc_range[0],
                    vmax=unc_range[1],
                )
                xlabel = (
                    "Stopped vehicle position"
                    if scenario == "standstill"
                    else "Fast vehicle speed"
                )
                ax.set_xlabel(xlabel, fontsize=20)
                ax.set_ylabel("Step", fontsize=20)
                ax.set_title(scenario, fontsize=20)
                ax.tick_params(labelrotation=45)
                ax.figure.axes[-1].yaxis.label.set_size(20)

                xticks = ax.xaxis.get_major_ticks()
                for i in range(len(xticks) // 2):
                    xticks[2 * i + 1].set_visible(False)

                # plt.show()
                plt.savefig(f"./videos/{model_name}_{scenario}_v3.png")
                plt.close()


if __name__ == "__main__":
    models = [
        {
            "name": "Ensemble RPF DQN",
            "color": "red",
            "train": {
                "show": True,
                "paths": [
                    "./logs/train_agent_20230715_211722_rpf_v14/data.hdf5",
                ],
                "show_uncertainty": True,
                "model_uncertainty": 0,
            },
            "base_path": "./logs/train_agent_20230715_211722_rpf_v14/",

            "ROC": {
                "use_uncertainty": True,
                "path": "rerun_test_scenarios_U_v5.csv",
                "mark": "o-",
            },

            "unc_heatmap": {
                "show": True,
                "fast_overtaking": "fast_overtaking_NU_v3.csv",
                "standstill": "standstill_NU_v3.csv",
                "unc_range": [0.02, 0.07],
            }
        },
        {
            "name": "DAE DQN K=1",
            "color": "green",
            "train": {
                "show": True,
                "paths": [
                    "./logs/train_agent_20231006_154948_dae_v5/data.hdf5",
                ],
                "show_uncertainty": True,
                "model_uncertainty": 0,
            },

            "base_path": "./logs/train_agent_20231006_154948_dae_v5/",

            "ROC": {
                "use_uncertainty": True,
                "path": "rerun_test_scenarios_U_v5.csv",
                "mark": "o-",
            },

            "unc_heatmap": {
                "show": True,
                "fast_overtaking": "fast_overtaking_NU_v3.csv",
                "standstill": "standstill_NU_v3.csv",
                "unc_range": [-60, 50],
            },
        },
        {
            "name": "DAE Ensemble RPF DQN K=1",
            "color": "orange",
            "train": {
                "show": True,
                "paths": [
                    "./logs/train_dae_rpf_agent_20240319_204634/data.hdf5",
                ],
                "show_uncertainty": True,
                "model_uncertainty": 0,
            },

            "base_path": "./logs/train_dae_rpf_agent_20240319_204634/",

            "ROC": {
                "use_uncertainty": True,
                "path": "rerun_test_scenarios_U_v5.csv",
                "mark": "o-",
            },

            "unc_heatmap": {
                "show": True,
                "fast_overtaking": "fast_overtaking_NU_v3.csv",
                "standstill": "standstill_NU_v3.csv",
                "unc_range": [97.7, 99],
            },
        },
        {
            "name": "Random DQN",
            "color": "magenta",
            "train": {
                "show": False,
            },

            "base_path": "./logs/random_agent/",

            "ROC": {
                "use_uncertainty": True,
                "path": "rerun_test_scenarios_U_v5.csv",
                "mark": "o-",
            },

            "unc_heatmap": {
                "show": False,
            },
        },
        {
            "name": "Standard DQN",
            "color": "blue",
            "train": {
                "show": True,
                "paths": [
                    "./logs/train_agent_20230323_235314_dqn_6M_v3/data.hdf5",
                ],
                "show_uncertainty": False,
                "model_uncertainty": 0,
            },

            "base_path": "./logs/train_agent_20230323_235314_dqn_6M_v3/",

            "ROC": {
                "use_uncertainty": False,
                "path": "rerun_test_scenarios_NU_v5.csv",
                "mark": ".",
            },

            "unc_heatmap": {
                "show": False,
            },
        },
    ]

    plot_train()
    # plot_rerun_test_v3()
    plot_tests_v3()
