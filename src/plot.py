import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.ndimage import gaussian_filter1d

PERCENTILE_DOWN = 1
PERCENTILE_UP = 99
EPSILON = 1e-15

def read_file(path):
    steps = []

    uncertainty = []
    uncertainty_up = []
    uncertainty_low = []
    uncertainty_std = []

    collision_rate = []

    collision_speed = []

    rewards = []
    with h5py.File(path, "r") as f:
        for step_key, step in f.items():
            steps.append(int(step_key))

            unc, rew, col, col_speed, nb_steps = step["uncertainties"][()], step["reward"][()], step["collision"][()], step["collision_speed"][()], step["steps"][()]

            total_steps = np.sum(nb_steps)
            unc = np.abs(unc) / total_steps
            
            uncertainty.append(np.mean(unc))
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
    uncertainty = (np.array(uncertainty, dtype=np.float16) + EPSILON)
    uncertainty_up = np.array(uncertainty_up, dtype=np.float16)
    uncertainty_low = np.array(uncertainty_low, dtype=np.float16)
    uncertainty_std = np.array(uncertainty_std, dtype=np.float16)
    collision_rate = np.array(collision_rate, dtype=np.float16)
    collision_speed = np.array(collision_speed, dtype=np.float16)
    rewards = np.array(rewards, dtype=np.float16)

    arr1inds = steps.argsort()
    steps = steps[arr1inds]
    uncertainty = uncertainty[arr1inds]
    uncertainty_up = uncertainty_up[arr1inds]
    uncertainty_low = uncertainty_low[arr1inds]
    uncertainty_std = uncertainty_std[arr1inds]
    collision_rate = collision_rate[arr1inds]
    collision_speed = collision_speed[arr1inds]
    rewards = rewards[arr1inds]

    return steps, (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std), (collision_rate, collision_speed), rewards

if __name__ == "__main__":
    models = [
        {
            "paths": [
                './logs/train_agent_20221123_195838/data.hdf5'
            ],
            "name": "Ensemble RPF DQN",
            "show_uncertainty": True,
            "color": "red",
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
            label="Std",
        )
        ax2.spines["top"].set_visible(False)
        plt.xlabel('Traning step', fontsize=14)
        plt.ylabel('Normalized Reward', fontsize=14)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # Uncertainty
    ax1 = plt.subplot(221, sharex = ax2)
    for model in models:
        if model["show_uncertainty"]:
            steps, (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std), _, _ = read_file(model["paths"][MODEL_NB])
            # print(f"Max: {np.max(uncertainty)}\tMin: {np.min(uncertainty)}\tMean: {np.mean(uncertainty)}\t Std: {np.std(uncertainty)}")
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
            plt.ylim((0,0.5))
            # plt.yscale('log')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    

    # Collision rate
    ax3 = plt.subplot(223, sharex = ax2)
    for model in models:
        steps, _, (collision_rate, _), _ = read_file(model["paths"][MODEL_NB])
        plt.plot(steps, 1 - collision_rate, color=model["color"], label="Mean {}".format(model["name"]))
        plt.xlabel('Traning step', fontsize=14)
        plt.ylabel('Collision Free Episodes', fontsize=14)
        plt.legend()
        ax3.spines["top"].set_visible(False)
        plt.ylim((-0.05,1.05))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    

    # Collision Speed
    ax4 = plt.subplot(224, sharex = ax2)
    for model in models:
        steps, _, (_, collision_speed), _ = read_file(model["paths"][MODEL_NB])
        plt.plot(steps, collision_speed, color=model["color"], label="Mean {}".format(model["name"]), alpha=0.1)
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

    plt.show()
    #plt.savefig('./logs/fig.png')
    #plt.close()
    