import h5py
import numpy as np

PERCENTILE_DOWN = 1
PERCENTILE_UP = 99
EPSILON = 1e-15

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
            unc = np.abs(unc / total_steps)

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
            unc = np.abs(unc / total_steps)
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


if __name__ == "__main__":
    step, _, _, reward = read_file("./logs/train_agent_20230323_235219_rpf_6M_v3/data.hdf5")
