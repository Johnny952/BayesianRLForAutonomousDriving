import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    steps = []

    with h5py.File(path, "r") as file:
        for step_key, step in file.items():
            steps.append(int(step_key))
            
    steps = np.array(steps)
    max_step = np.max(steps)

    uncertainties = np.array([])
    with h5py.File(path, "r") as file:
        s = file[str(max_step)]["uncertainties_history"]
        for unc_key, unc in s.items():
            uncertainties = np.concatenate((uncertainties, np.array(unc[()])))

    return (
        steps,
        uncertainties,
    )

def read_file2(path, ep_type=int):
    thresholds = []
    episodes = []
    uncert = np.array([])
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            thresholds.append(float(row[0]))
            episodes.append(ep_type(row[1]))
            uncert = np.concatenate((uncert, np.array([np.abs(float(d)) for d in row[2:]])))
    return thresholds, episodes, uncert

def plot_distributions(models, mode="hdf5"):
    for model in models:
        if mode == "hdf5":
            _, uncertainties = read_file(model["path"])
        elif mode == "csv":
            _, _, uncertainties = read_file2(model["csv"])
        else:
            raise Exception
        bins = model["bins"]
        range_ = model["range"]
        marks = model["custom_marks"]
        plt.figure()
        plt.hist(uncertainties, bins=bins, range=range_)
        plt.xlabel("Uncertainty")
        model_name = model["name"]

        mean_ = np.mean(uncertainties)
        std_ = np.std(uncertainties)
        print(model_name)
        print(mean_ + std_)
        print(mean_ + 2*std_, '\n')

        plt.axvline(x=mean_ + std_, color='red', linestyle='dashed', label=r'$\mu+\sigma$: {:.3f}'.format(mean_ + std_))
        plt.axvline(x=mean_ + 2*std_, color='green', linestyle='dashed', label=r'$\mu+2\sigma$: {:.3f}'.format(mean_ + 2*std_))

        for m in marks:
            plt.axvline(x=m, color='black', linestyle='dashed', label="Custom: {}".format(m))

        plt.title(model_name)
        plt.legend()
        plt.savefig(f"./videos/dist_{model_name}.png")
        plt.close()

if __name__ == "__main__":
    models = [
        {
            "name": "Ensemble RPF DQN",
            "path": "./logs/train_agent_20230628_172622_rpf_v10/data.hdf5",
            "csv": "./logs/train_agent_20230628_172622_rpf_v10/rerun_test_scenarios_NU_uncerts.csv",
            "custom_marks": [],
            "bins": 50,
            "range": (0, 0.035),
        },
        {
            "name": "DAE DQN",
            "path": "./logs/train_agent_20230628_172734_ae_v10/data.hdf5",
            "csv": "./logs/train_agent_20230628_172734_ae_v10/rerun_test_scenarios_NU_uncerts.csv",
            "custom_marks": [87],
            "bins": 40,
            "range": (65, 110),
        }
    ]

    plot_distributions(models, mode="csv")