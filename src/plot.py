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
            ) = read_file(model["paths"][model_nb])
            plt.plot(
                steps,
                uncertainty,
                color=model["color"],
                label=f"Mean {model_name}",
            )
            plt.fill_between(
                steps,
                (uncertainty - uncertainty_std),
                (uncertainty + uncertainty_std),
                color=model["color"],
                alpha=0.2,
                label="Std",
            )
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
    plt.ylim((0, 1))
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
    plt.savefig("./videos/train.png")
    plt.close()

    # for scenario in '':
    #     fig, axs = plt.subplots(ncols=2, nrows=2)
    #     fig.set_figwidth(16)
    #     fig.set_figheight(16)
    #     ax1 = axs[0, 0]
    #     ax2 = axs[0, 1]
    #     ax3 = axs[1, 0]
    #     ax4 = axs[1, 1]

    #     for model in models:
    #         base_path = model["multiple_test"]["base_path"]
    #         model_name = model["name"]
    #         if model["multiple_test"][scenario]["u"]:
    #             (
    #                 _,
    #                 rewards,
    #                 collision_rates,
    #                 nb_safe_actions,
    #                 nb_safe_action_hard,
    #                 collision_speeds,
    #             ) = read_test(f"{base_path}{scenario}_U.csv")
    #             _, [filtered_rates, filtered_rewards] = collapse_duplicated(
    #                 collision_rates, rewards
    #             )
    #             ax1.plot(
    #                 filtered_rates,
    #                 filtered_rewards,
    #                 ".-",
    #                 color=model["color"],
    #                 label=f"{model_name} U",
    #                 alpha=1,
    #             )

    #             _, [
    #                 filtered_safe_action,
    #                 filtered_rates,
    #                 filtered_rewards,
    #                 filtered_speeds,
    #             ] = collapse_duplicated(
    #                 nb_safe_actions, collision_rates, rewards, collision_speeds
    #             )
    #             ax2.plot(
    #                 filtered_safe_action,
    #                 filtered_rewards,
    #                 ".-",
    #                 color=model["color"],
    #                 label=f"{model_name} U",
    #                 alpha=1,
    #             )
    #             ax3.plot(
    #                 filtered_safe_action,
    #                 filtered_rates,
    #                 ".-",
    #                 color=model["color"],
    #                 label=f"{model_name} U",
    #                 alpha=1,
    #             )
    #             ax4.plot(
    #                 filtered_safe_action,
    #                 filtered_speeds,
    #                 ".-",
    #                 color=model["color"],
    #                 label=f"{model_name} U",
    #                 alpha=1,
    #             )

    #         if model["multiple_test"][scenario]["nu"]:
    #             (
    #                 _,
    #                 rewards,
    #                 collision_rates,
    #                 _,
    #                 _,
    #                 collision_speeds,
    #             ) = read_test(f"{base_path}{scenario}_NU.csv")
    #             ax1.plot(
    #                 collision_rates,
    #                 rewards,
    #                 ".",
    #                 color=model["color"],
    #                 label=f"{model_name} NU",
    #                 alpha=1,
    #                 markersize=14,
    #             )
    #             ax2.axhline(
    #                 y=rewards[0],
    #                 xmin=0.0,
    #                 xmax=1.0,
    #                 color=model["color"],
    #                 linestyle="--",
    #             )
    #             ax3.axhline(
    #                 y=collision_rates[0],
    #                 xmin=0.0,
    #                 xmax=1.0,
    #                 color=model["color"],
    #                 linestyle="--",
    #             )
    #             ax4.axhline(
    #                 y=collision_speeds[0],
    #                 xmin=0.0,
    #                 xmax=1.0,
    #                 color=model["color"],
    #                 linestyle="--",
    #             )

    #     plt.suptitle(f"{scenario}", fontsize=25)
    #     ax1.set_xlim(left=0)
    #     # ax1.set_ylim(bottom=-4)
    #     ax1.set_ylabel("Rewards", fontsize=16)

    #     # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #     ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    #     ax1.set_xlabel("Collision Rate", fontsize=16)
    #     ax1.legend()
    #     ax1.grid()

    #     # ax2.set_ylim(bottom=-4)
    #     ax2.set_xlim(left=0, right=1)
    #     ax2.set_ylabel("Reward", fontsize=16)
    #     ax2.set_xlabel("Number Safe actions", fontsize=16)

    #     ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    #     ax3.set_ylim(bottom=0)
    #     ax3.set_xlim(left=0, right=1)
    #     ax3.set_ylabel("Collision Rate", fontsize=16)
    #     ax3.set_xlabel("Number Safe actions", fontsize=16)

    #     ax4.set_ylim(bottom=0)
    #     ax4.set_xlim(left=0, right=1)
    #     ax4.legend()
    #     ax4.set_ylabel("Collision Speeds", fontsize=16)
    #     ax4.set_xlabel("Number Safe actions", fontsize=16)

    #     # plt.show()
    #     plt.savefig(f"./videos/{scenario}.png")
    #     plt.close()


def plot_tests2():
    for model in models:
        plt.figure()
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.set_figwidth(10)
        fig.set_figheight(5)
        plots = {
            "standstill": axs[0],
            "fast_overtaking": axs[1],
        }
        max_plots = model["tests_plots"]
        for scenario in model["tests"].keys():
            filepath = model["tests"][scenario]
            if filepath:
                _, _, uncerts = read_test2(filepath)
                for i, uncert in enumerate(uncerts[:max_plots]):
                    plots[scenario].plot(uncert, label=f"Run {str(i+1)}")
        model_name = model["name"]
        plt.suptitle(f"Model {model_name}")

        for scenario, ax in plots.items():
            ax.set_ylabel("Uncertainty")
            ax.set_xlabel(" Timestamp")
            ax.set_title(scenario)
        axs[0].legend()

        plt.savefig(f"./videos/{model_name}.png")
        plt.close()


def plot_rerun_test_v3():
    scenario = "rerun_test_scenarios"
    fig1, ax1 = plt.subplots(ncols=1, nrows=1)
    fig1.set_figwidth(16)
    fig1.set_figheight(16)

    fig2, ax2 = plt.subplots(ncols=1, nrows=1)
    fig2.set_figwidth(16)
    fig2.set_figheight(16)

    fig3, ax3 = plt.subplots(ncols=1, nrows=1)
    fig3.set_figwidth(16)
    fig3.set_figheight(16)

    fig4, ax4 = plt.subplots(ncols=1, nrows=1)
    fig4.set_figwidth(16)
    fig4.set_figheight(16)

    for model in models:
        base_path = model["test_v3"]["base_path"]
        model_name = model["name"]
        sufix = model["test_v3"]["rerun_sufix"]
        mark = model["test_v3"]["mark"]
        second_mark = model["test_v3"]["second_mark"]
        use_v0 = model["test_v3"]["paths"][scenario]["v0"]
        if model["test_v3"]["paths"][scenario]["u"]:
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
            ) = read_test(f"{base_path}{scenario}_U{sufix}.csv")
            print(model_name)
            print("Threshold\tReward\tSafe Actions\tCollision Rates")
            for i in range(len(thresholds)):
                print(
                    "{:.3f}\t\t{:.3f}\t{:.2f}\t\t\t{:.2f}%".format(
                        thresholds[i],
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

            if use_v0:
                (
                    _,
                    rewards_v0,
                    collision_rates_v0,
                    nb_safe_actions_v0,
                    _,
                    collision_speeds_v0,
                    _,
                    _,
                    _,
                ) = read_test(f"{base_path}{scenario}_U_v0.csv")
                # _, [filtered_rates_s_v0, filtered_rewards_s_v0] = collapse_duplicated(collision_rates_v0, rewards_v0)
                _, [filtered_rates_s_v0, filtered_rewards_s_v0] = sort_duplicated(
                    collision_rates_v0, rewards_v0
                )
                # filtered_rates_s_v0, filtered_rewards_s_v0 = collision_rates_v0, rewards_v0
                _, [
                    filtered_safe_action_v0,
                    filtered_rates_v0,
                    filtered_rewards_v0,
                    filtered_speeds_v0,
                ] = collapse_duplicated(
                    nb_safe_actions_v0,
                    collision_rates_v0,
                    rewards_v0,
                    collision_speeds_v0,
                    reduction=np.mean,
                )
            ax1.plot(
                filtered_rates_s,
                filtered_rewards_s,
                mark,
                color=model["color"],
                label=f"{model_name} U test v1",
                alpha=1,
                linewidth=2,
            )

            ax2.plot(
                filtered_safe_action,
                filtered_rewards,
                mark,
                color=model["color"],
                label=f"{model_name} U test v1",
                alpha=1,
            )
            ax3.plot(
                filtered_safe_action,
                filtered_rates,
                mark,
                color=model["color"],
                label=f"{model_name} U test v1",
                alpha=1,
            )
            ax4.plot(
                filtered_safe_action,
                filtered_speeds,
                mark,
                color=model["color"],
                label=f"{model_name} U",
                alpha=1,
            )

            if use_v0:
                ax1.plot(
                    filtered_rates_s_v0,
                    filtered_rewards_s_v0,
                    second_mark,
                    color=model["color"],
                    label=f"{model_name} U test v0",
                    alpha=1,
                )
                ax2.plot(
                    filtered_safe_action_v0,
                    filtered_rewards_v0,
                    second_mark,
                    color=model["color"],
                    label=f"{model_name} U test v0",
                    alpha=1,
                )
                ax3.plot(
                    filtered_safe_action_v0,
                    filtered_rates_v0,
                    second_mark,
                    color=model["color"],
                    label=f"{model_name} U test v0",
                    alpha=1,
                )
                ax4.plot(
                    filtered_safe_action_v0,
                    filtered_speeds_v0,
                    second_mark,
                    color=model["color"],
                    label=f"{model_name} U",
                    alpha=1,
                )

        if model["test_v3"]["paths"][scenario]["nu"]:
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
            ) = read_test(f"{base_path}{scenario}_NU{sufix}.csv")
            if use_v0:
                (
                    _,
                    rewards_v0,
                    collision_rates_v0,
                    _,
                    _,
                    collision_speeds_v0,
                    _,
                    _,
                    _,
                ) = read_test(f"{base_path}{scenario}_NU_v0.csv")
            ax1.plot(
                collision_rates,
                rewards,
                mark,
                color=model["color"],
                label=f"{model_name} NU test v1",
                alpha=1,
                markersize=16,
            )
            ax2.axhline(
                y=rewards[0],
                xmin=0.0,
                label=f"{model_name} NU test v1",
                color=model["color"],
                linestyle="--",
            )
            ax3.axhline(
                y=collision_rates[0],
                xmin=0.0,
                label=f"{model_name} NU test v1",
                color=model["color"],
                linestyle="--",
            )
            ax4.axhline(
                y=collision_speeds[0],
                xmin=0.0,
                label=f"{model_name} NU",
                color=model["color"],
                linestyle="--",
            )

            if use_v0:
                ax1.plot(
                    collision_rates_v0,
                    rewards_v0,
                    second_mark,
                    color=model["color"],
                    label=f"{model_name} NU test v0",
                    alpha=1,
                )
                ax2.axhline(
                    y=rewards_v0[0], xmin=0.0, color=model["color"], linestyle="-."
                )
                ax3.axhline(
                    y=collision_rates_v0[0],
                    xmin=0.0,
                    color=model["color"],
                    linestyle="-.",
                )
                ax4.axhline(
                    y=collision_speeds_v0[0],
                    xmin=0.0,
                    color=model["color"],
                    linestyle="-.",
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
    ax2.set_xlabel("Number Safe actions", fontsize=16)

    ax3.set_title(f"{scenario}", fontsize=25)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax3.set_ylim(bottom=0)
    ax3.legend()
    ax3.set_xlim(left=0)
    ax3.set_ylabel("Collision Rate", fontsize=16)
    ax3.set_xlabel("Number Safe actions", fontsize=16)

    ax4.set_ylim(bottom=0)
    ax4.set_xlim(left=0)
    ax4.legend()
    ax4.set_ylabel("Collision Speeds", fontsize=16)
    ax4.set_xlabel("Number Safe actions", fontsize=16)

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
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.set_figwidth(16)
        fig.set_figheight(8)

        base_path = model["test_v3"]["base_path"]
        model_name = model["name"]
        sufix = model["test_v3"]["sufix"]
        paths = model["test_v3"]["paths"]

        fig.suptitle(model_name)

        for idx, scenario in enumerate(
            ["standstill", "fast_overtaking"]
        ):  # standstill, "fast_overtaking"
            if paths[scenario]["nu"]:
                path = f"{base_path}{scenario}_NU{sufix}.csv"
                _, pos_vel, unc = read_test2(path, ep_type=float)
                max_len = len(max(unc, key=len))
                min_unc = min(unc)
                flat_unc = []
                for u in unc:
                    flat_unc += u
                print(model_name, np.min(flat_unc), np.max(flat_unc))
                unc_range = model["test_v3"]["unc_range"]
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
                    ax=axs[ax_idx],
                    cbar_kws={"label": "uncertainty"},
                    vmin=unc_range[0],
                    vmax=unc_range[1],
                )
                xlabel = (
                    "Stopped vehicle position"
                    if scenario == "standstill"
                    else "Fast vehicle speed"
                )
                axs[ax_idx].set_xlabel(xlabel, fontsize=16)
                axs[ax_idx].set_ylabel("Step", fontsize=16)
                axs[ax_idx].set_title(scenario, fontsize=16)
                axs[ax_idx].tick_params(labelrotation=45)

                xticks = axs[ax_idx].xaxis.get_major_ticks()
                for i in range(len(xticks) // 2):
                    xticks[2 * i + 1].set_visible(False)

        # plt.show()
        plt.savefig(f"./videos/{model_name}_v3.png")
        plt.close()


if __name__ == "__main__":
    models = [
        {
            "paths": [
                "./logs/train_agent_20230715_211722_rpf_v14/data.hdf5",
            ],
            # "multiple_test": {
            #     "base_path": "./logs/train_agent_20230715_211722_rpf_v14/",
            #     "rerun_test_scenarios": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "standstill": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "fast_overtaking": {
            #         "u": True,
            #         "nu": True,
            #     },
            # },
            "name": "Ensemble RPF DQN",
            "show_uncertainty": True,
            "color": "red",
            # "tests": {
            #     "rerun_test_scenarios": None,
            #     "standstill": "./logs/train_agent_20230715_211722_rpf_v14/standstill_NU_2.csv",
            #     "fast_overtaking": "./logs/train_agent_20230715_211722_rpf_v14/fast_overtaking_NU_2.csv",
            # },
            # "tests_plots": 5,
            "test_v3": {
                "sufix": "_v3",
                "rerun_sufix": "_v5",
                "mark": "X-",
                "second_mark": "D-",
                "base_path": "./logs/train_agent_20230715_211722_rpf_v14/",
                "unc_range": [0.012, 0.08],
                "paths": {
                    "rerun_test_scenarios": {
                        "u": True,
                        "nu": False,
                        "v0": False,
                    },
                    "standstill": {
                        "u": False,
                        "nu": True,
                    },
                    "fast_overtaking": {
                        "u": False,
                        "nu": True,
                    },
                },
            },
        },
        {
            "paths": [
                "./logs/train_agent_20230815_160313_ae_v15/data.hdf5",
            ],
            # "multiple_test": {
            #     "base_path": "./logs/train_agent_20230815_160313_ae_v15/",
            #     "rerun_test_scenarios": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "standstill": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "fast_overtaking": {
            #         "u": True,
            #         "nu": True,
            #     },
            # },
            "name": "AE DQN",
            "show_uncertainty": True,
            "color": "green",
            # "tests": {
            #     "rerun_test_scenarios": None,
            #     "standstill": "./logs/train_agent_20230815_160313_ae_v15/standstill_NU_2.csv",
            #     "fast_overtaking": "./logs/train_agent_20230815_160313_ae_v15/fast_overtaking_NU_2.csv",
            # },
            # "tests_plots": 5,
            "test_v3": {
                "sufix": "_v3",
                "rerun_sufix": "_v5",
                "mark": "v-",
                "second_mark": "^-",
                "base_path": "./logs/train_agent_20230815_160313_ae_v15/",
                "unc_range": [-200, 250],
                "paths": {
                    "rerun_test_scenarios": {
                        "u": True,
                        "nu": False,
                        "v0": False,
                    },
                    "standstill": {
                        "u": False,
                        "nu": True,
                    },
                    "fast_overtaking": {
                        "u": False,
                        "nu": True,
                    },
                },
            },
        },
        {
            "paths": [
                "./logs/train_agent2_20230903_214928_ae_v22_3/data.hdf5",
            ],
            # "multiple_test": {
            #     "base_path": "./logs/train_agent2_20230903_214928_ae_v22_3/",
            #     "rerun_test_scenarios": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "standstill": {
            #         "u": True,
            #         "nu": True,
            #     },
            #     "fast_overtaking": {
            #         "u": True,
            #         "nu": True,
            #     },
            # },
            "name": "AE DQN 2",
            "show_uncertainty": False,
            "color": "magenta",
            # "tests": {
            #     "rerun_test_scenarios": None,
            #     "standstill": "./logs/train_agent2_20230903_214928_ae_v22_3/standstill_NU_2.csv",
            #     "fast_overtaking": "./logs/train_agent2_20230903_214928_ae_v22_3/fast_overtaking_NU_2.csv",
            # },
            # "tests_plots": 5,
            "test_v3": {
                "sufix": "_v3",
                "rerun_sufix": "_v5",
                "mark": "v-",
                "second_mark": "^-",
                "base_path": "./logs/train_agent2_20230903_214928_ae_v22_3/",
                "unc_range": [None, None],
                "paths": {
                    "rerun_test_scenarios": {
                        "u": True,
                        "nu": False,
                        "v0": False,
                    },
                    "standstill": {
                        "u": False,
                        "nu": True,
                    },
                    "fast_overtaking": {
                        "u": False,
                        "nu": True,
                    },
                },
            },
        },
        {
            "paths": [
                "./logs/train_agent_20230323_235314_dqn_6M_v3/data.hdf5",
            ],
            # "multiple_test": {
            #     "base_path": "./logs/train_agent_20230323_235314_dqn_6M_v3/",
            #     "rerun_test_scenarios": {
            #         "u": False,
            #         "nu": True,
            #     },
            #     "standstill": {
            #         "u": False,
            #         "nu": True,
            #     },
            #     "fast_overtaking": {
            #         "u": False,
            #         "nu": True,
            #     },
            # },
            "name": "Standard DQN",
            "show_uncertainty": False,
            "color": "blue",
            # "tests": {
            #     "rerun_test_scenarios": None,
            #     "standstill": None,
            #     "fast_overtaking": None,
            # },
            # "tests_plots": 5,
            "test_v3": {
                "sufix": "_v5",
                "rerun_sufix": "_v5",
                "mark": ".",
                "second_mark": "*",
                "base_path": "./logs/train_agent_20230323_235314_dqn_6M_v3/",
                "paths": {
                    "rerun_test_scenarios": {
                        "u": False,
                        "nu": True,
                        "v0": False,
                    },
                    "standstill": {
                        "u": False,
                        "nu": False,
                    },
                    "fast_overtaking": {
                        "u": False,
                        "nu": False,
                    },
                },
            },
        },
    ]

    # plot_train()
    #### plot_tests2()
    # plot_rerun_test_v3()
    plot_tests_v3()
