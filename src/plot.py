import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FormatStrFormatter

PERCENTILE_DOWN = 1
PERCENTILE_UP = 99
EPSILON = 1e-15

def read_test(path):
    thresholds = []
    rewards = []
    collision_rates = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            thresholds.append(float(row[0]))
            rewards.append(float(row[1]))
            collision_rates.append(float(row[2]))
    return np.array(thresholds), np.array(rewards), np.array(collision_rates)


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
    with h5py.File(path, "r") as f:
        for step_key, step in f.items():
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
    with h5py.File(path, "r") as f:
        for step_key, step in f.items():
            steps.append(int(step_key))

            unc, rew, col, col_speed, nb_steps = step["uncertainties"][()], step["reward"][()], step["collision"][()], step["collision_speed"][()], step["steps"][()]
            total_steps = np.sum(nb_steps)
            unc = unc / total_steps
            if negative_unc:
                unc = -unc

            if unc_normalized and max_unc != min_unc:
                unc = (unc - min_unc) / (max_unc - min_unc + EPSILON)
                # unc = (unc - mean_min) / (mean_max - mean_min + EPSILON)

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
            if (len(coll_s) == 0):
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

    return steps, (uncertainty_mean, uncertainty_up, uncertainty_low, uncertainty_std), (collision_rate, collision_speed), rewards

if __name__ == "__main__":
    models = [
        {
            "paths": [
                # './logs/rpf_train_agent_20230127_221001_this/data.hdf5',
                # './logs/rpf_train_agent_20230130_210919/data.hdf5',
                # './logs/rpf_train_agent_20230202_154753/data.hdf5',
                # './logs/rpf_train_agent_20230204_214216/data.hdf5',
                # './logs/rpf_train_agent_20230210_195119/data.hdf5',
                # './logs/rpf_train_agent_20230210_195214/data.hdf5',
                # './logs/rpf_train_agent_20230213_211943/data.hdf5',
                # './logs/rpf_train_agent_20230213_211945/data.hdf5',
                # './logs/rpf_train_agent_20230217_173810/data.hdf5',
                './logs/train_agent_20230323_235219_rpf_6M_v3/data.hdf5',
            ],
            'multiple_test': {
                'rerun_test_scenarios': './logs/train_agent_20230323_235219_rpf_6M_v3/rerun_test_scenarios.csv',
                'standstill': None,
                'fast_overtaking': None,
            },
            "name": "Ensemble RPF DQN",
            "show_uncertainty": True,
            "negative_uncertainty": False,
            "color": "red",
        },
        {
            "paths": [
                # './logs/dqn_train_agent_20230127_221037/data.hdf5',
                # './logs/dqn_train_agent_20230130_210827/data.hdf5',
                # './logs/dqn_train_agent_20230131_212058/data.hdf5',
                # './logs/dqn_train_agent_20230201_210132/data.hdf5',
                # './logs/dqn_train_agent_20230202_154709/data.hdf5',
                # './logs/dqn_train_agent_20230204_214252/data.hdf5',
                # './logs/dqn_train_agent_20230205_230839/data.hdf5',
                # './logs/dqn_train_agent_20230207_031945_this/data.hdf5',
                './logs/train_agent_20230323_235314_dqn_6M_v3/data.hdf5',
            ],
            'multiple_test': {
                'rerun_test_scenarios': None,
                'standstill': None,
                'fast_overtaking': None,
            },
            "name": "Standard DQN",
            "show_uncertainty": False,
            "negative_uncertainty": False,
            "color": "blue",
        },
        {
            "paths": [
                # './logs/bnn_train_agent_20230210_195642/data.hdf5',
                # './logs/train_agent_20230220_205020/data.hdf5',
                # './logs/train_agent_20230220_205123/data.hdf5',
                # './logs/train_agent_20230222_234907/data.hdf5',
                './logs/train_agent_20230329_191111_bnn_6M_v3/data.hdf5',
            ],
            'multiple_test': {
                'rerun_test_scenarios': None,#'./logs/train_agent_20230329_191111_bnn_6M_v3/rerun_test_scenarios.csv',
                'standstill': None,
                'fast_overtaking': None,
            },
            "name": "BNN DQN",
            "show_uncertainty": True,
            "negative_uncertainty": False,
            "color": "orange",
        },
        {
            "paths": [
                # './logs/bnn_train_agent_20230210_195642/data.hdf5',
                # './logs/train_agent_20230220_205020/data.hdf5',
                # './logs/train_agent_20230220_205123/data.hdf5',
                # './logs/train_agent_20230222_234907/data.hdf5',
                './logs/train_agent_20230327_142839_ae_6M_v3/data.hdf5',
            ],
            'multiple_test': {
                'rerun_test_scenarios': None,#'./logs/train_agent_20230327_142839_ae_6M_v3/rerun_test_scenarios.csv',
                'standstill': None,
                'fast_overtaking': None,
            },
            "name": "AE DQN",
            "show_uncertainty": True,
            "negative_uncertainty": True,
            "color": "green",
        },
    ]
    MODEL_NB = 0

    plt.figure(figsize=(13, 8))

    # Rewards
    ax2 = plt.subplot(222)
    for model in models:
        rewards = []
        steps = []
        for path in model["paths"]:
            s, _, _, r = read_file(path)
            rewards.append(r)
            steps.append(s)
        rewards = np.array(rewards)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        steps = steps[0]

        plt.plot(steps, rewards_mean, color=model["color"], label="Mean {}".format(model["name"]))
        plt.fill_between(
            steps,
            (rewards_mean - rewards_std),
            (rewards_mean + rewards_std),
            color=model["color"],
            alpha=0.2,
            #label="Std",
        )
    ax2.spines["top"].set_visible(False)
    plt.xlabel('Traning step', fontsize=14)
    plt.ylabel('Normalized Reward', fontsize=14)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.legend()

    # Uncertainty
    ax1 = plt.subplot(221, sharex = ax2)
    for model in models:
        if model["show_uncertainty"]:
            steps, (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std), _, _ = read_file(model["paths"][MODEL_NB], negative_unc=model["negative_uncertainty"])
            plt.plot(steps, uncertainty, color=model["color"], label="Mean {}".format(model["name"]))
            # plt.fill_between(
            #     steps,
            #     (uncertainty - uncertainty_std),
            #     (uncertainty + uncertainty_std),
            #     color=model["color"],
            #     alpha=0.2,
            #     label="Std",
            # )
            plt.fill_between(
                steps,
                (uncertainty_up),
                (uncertainty_low),
                color=model["color"],
                alpha=0.1,
            )
    ax1.spines["top"].set_visible(False)
    plt.xlabel('Traning step', fontsize=14)
    plt.ylabel('Normalized Uncertainty', fontsize=14)
    # plt.ylim((0,0.1))
    # plt.yscale('log')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # Collision rate
    ax3 = plt.subplot(223, sharex = ax2)
    for model in models:
        steps, _, (collision_rate, _), _ = read_file(model["paths"][MODEL_NB])
        plt.plot(steps, 1 - collision_rate, color=model["color"], label="Mean {}".format(model["name"]))
    plt.xlabel('Traning step', fontsize=14)
    plt.ylabel('Collision Free Episodes', fontsize=14)
    ax3.spines["top"].set_visible(False)
    plt.ylim((-0.05,1.05))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    

    # Collision Speed
    ax4 = plt.subplot(224, sharex = ax2)
    for model in models:
        steps, _, (_, collision_speed), _ = read_file(model["paths"][MODEL_NB])
        plt.plot(steps, collision_speed, color=model["color"], label="Mean {}".format(model["name"]), alpha=.1)
        filtered_speeds = gaussian_filter1d(collision_speed.astype(np.float32), sigma=2)
        plt.plot(steps, filtered_speeds, color=model["color"], label="Mean {}".format(model["name"]), alpha=1)
        # plt.fill_between(
        #     steps,
        #     (collision_speed_up),
        #     (collision_speed_low),
        #     color=model["color"],
        #     alpha=0.1,
        # )
    plt.xlabel('Traning step', fontsize=14)
    plt.ylabel('Collision Speed', fontsize=14)
    ax4.spines["top"].set_visible(False)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # plt.show()
    plt.savefig('./videos/fig.png')
    plt.close()
    
    # Rewards vs collision rate
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(13)
    index = -1
    scenario = 'rerun_test_scenarios'
    for model in models:
        if model['multiple_test'][scenario] is not None:
            thresholds, rewards, collision_rates = read_test(model['multiple_test'][scenario])
            idc = np.argsort(collision_rates)
            sorted_rates, sorted_rewards = collision_rates[idc], rewards[idc]
            # filtered_rewards = gaussian_filter1d(sorted_rewards.astype(np.float32), sigma=0.9)
            # plt.plot(sorted_rates, filtered_rewards, color=model["color"], label="{}".format(model["name"]), alpha=1)
            ax.plot(sorted_rates, sorted_rewards, color=model["color"], label=model["name"], alpha=1)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Collision Rate', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend()
    plt.grid()
    # plt.show()