import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def read_file(path):
    steps = []

    uncertainty = []
    uncertainty_up = []
    uncertainty_low = []
    uncertainty_std = []

    collision_rate = []

    collision_speed = []
    collision_speed_up = []
    collision_speed_low = []

    rewards = []
    rewards_up = []
    rewards_low = []
    max_unc = -1e10
    with h5py.File(path, "r") as f:
        for step_key, step in f.items():
            steps.append(int(step_key))

            unc, rew, col, col_speed, nb_steps = step["uncertainties"][()], step["reward"][()], step["collision"][()], step["collision_speed"][()], step["steps"][()]
            
            unc_ = unc / nb_steps
            
            
            max_unc = np.max(unc_) if np.max(unc_) > max_unc else max_unc
            uncertainty.append(np.mean(unc_))
            uncertainty_up.append(np.percentile(unc_, percentile_up))
            uncertainty_low.append(np.percentile(unc_, percentile_low))
            uncertainty_std.append(unc_.std())
            
            rewards.append(np.mean(rew / 100))
            rewards_up.append(np.percentile(rew / 100, percentile_up))
            rewards_low.append(np.percentile(rew / 100, percentile_low))
            
            rate = np.sum(col) / len(col)
            collision_rate.append(rate)

            coll_s = col_speed[col_speed > 0]
            if (len(coll_s) == 0):
                collision_speed.append(np.mean(0))
                collision_speed_up.append(0)
                collision_speed_low.append(0)
            else:
                collision_speed.append(np.mean(coll_s))
                collision_speed_up.append(np.percentile(coll_s, percentile_up))
                collision_speed_low.append(np.percentile(coll_s, percentile_low))
    
    steps = np.array(steps)
    max_unc = 1 if max_unc == 0 else max_unc
    uncertainty = (np.array(uncertainty, dtype=np.float16) + 1e-10) / max_unc
    uncertainty_up = np.array(uncertainty_up, dtype=np.float16) / max_unc
    uncertainty_low = np.array(uncertainty_low, dtype=np.float16) / max_unc
    uncertainty_std = np.array(uncertainty_std, dtype=np.float16)
    collision_rate = np.array(collision_rate, dtype=np.float16)
    collision_speed = np.array(collision_speed, dtype=np.float16)
    collision_speed_up = np.array(collision_speed_up, dtype=np.float16)
    collision_speed_low = np.array(collision_speed_low, dtype=np.float16)
    rewards = np.array(rewards, dtype=np.float16)
    rewards_up = np.array(rewards_up, dtype=np.float16)
    rewards_low = np.array(rewards_low, dtype=np.float16)

    arr1inds = steps.argsort()
    steps = steps[arr1inds[::-1]]
    uncertainty = uncertainty[arr1inds[::-1]]
    uncertainty_up = uncertainty_up[arr1inds[::-1]]
    uncertainty_low = uncertainty_low[arr1inds[::-1]]
    uncertainty_std = uncertainty_std[arr1inds[::-1]]
    collision_rate = collision_rate[arr1inds[::-1]]
    collision_speed = collision_speed[arr1inds[::-1]]
    collision_speed_up = collision_speed_up[arr1inds[::-1]]
    collision_speed_low = collision_speed_low[arr1inds[::-1]]
    rewards = rewards[arr1inds[::-1]]
    rewards_up = rewards_up[arr1inds[::-1]]
    rewards_low = rewards_low[arr1inds[::-1]]

    return steps, (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std), (collision_rate, collision_speed, collision_speed_up, collision_speed_low), (rewards, rewards_up, rewards_low)

if __name__ == "__main__":
    models = [
        {
            "path": './logs/rpf_20221123_195838/data.hdf5',
            "name": "Ensemble RPF DQN",
            "show_uncertainty": True,
            "color": "red",
        },
        {
            "path": './logs/dqn_20221115_200203/data.hdf5',
            "name": "Standar DQN",
            "show_uncertainty": False,
            "color": "blue",
        },
        {
            "path": './logs/bnn_broken_20221213_203952/data.hdf5',
            "name": "BNN DQN",
            "show_uncertainty": True,
            "color": "green",
        },
    ]

    percentile_low = 1
    percentile_up = 99

    plt.figure(figsize=(13, 8))

    # Rewards
    ax2 = plt.subplot(222)
    for model in models:
        steps, _, _, (rewards, rewards_up, rewards_low) = read_file(model["path"])
        plt.plot(steps, rewards, color=model["color"], label="Mean {}".format(model["name"]))
        plt.fill_between(
            steps,
            (rewards_up),
            (rewards_low),
            color=model["color"],
            alpha=0.1,
        )
        plt.xlabel('Traning step', fontsize=14)
        plt.ylabel('Normalized Reward', fontsize=14)
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # Uncertainty
    plt.subplot(221, sharex = ax2)
    for model in models:
        if model["show_uncertainty"]:
            steps, (uncertainty, uncertainty_up, uncertainty_low, uncertainty_std), _, _ = read_file(model["path"])
            plt.plot(steps, uncertainty, color=model["color"], label="Mean {}".format(model["name"]))
            # plt.fill_between(
            #     steps,
            #     (uncertainty - uncertainty_std),
            #     (uncertainty + uncertainty_std),
            #     color=color,
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
            plt.xlabel('Traning step', fontsize=14)
            plt.ylabel('Normalized Uncertainty', fontsize=14)
            plt.grid(True)
            plt.yscale('log')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # Collision rate
    plt.subplot(223, sharex = ax2)
    for model in models:
        steps, _, (collision_rate, _, _, _), _ = read_file(model["path"])
        plt.plot(steps, collision_rate, color=model["color"], label="Mean {}".format(model["name"]))
        plt.xlabel('Traning step', fontsize=14)
        plt.ylabel('Collision Rate', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # Collision Speed
    plt.subplot(224, sharex = ax2)
    for model in models:
        steps, _, (_, collision_speed, collision_speed_up, collision_speed_low), _ = read_file(model["path"])
        plt.plot(steps, collision_speed, color=model["color"], label="Mean {}".format(model["name"]))
        plt.fill_between(
            steps,
            (collision_speed_up),
            (collision_speed_low),
            color=model["color"],
            alpha=0.1,
        )
        plt.xlabel('Traning step', fontsize=14)
        plt.ylabel('Collision Speed', fontsize=14)
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    plt.savefig('./logs/fig.png')
    plt.close()
    