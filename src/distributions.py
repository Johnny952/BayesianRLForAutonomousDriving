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
            uncert = np.concatenate((uncert, np.array([float(d) for d in row[2:]])))
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
        perc95 = np.percentile(uncertainties, 95)
        perc975 = np.percentile(uncertainties, 97.5)
        perc99 = np.percentile(uncertainties, 99)
        perc995 = np.percentile(uncertainties, 99.5)
        perc999 = np.percentile(uncertainties, 99.9)
        print(model_name)
        print(r'$\mu+\sigma$: {}'.format(mean_ + std_))
        print(r'$\mu+2\sigma$: {}'.format(mean_ + 2 * std_))
        print("95%: {}".format(perc95))
        print("97.5%: {}".format(perc975))
        print("99%: {}".format(perc99))
        print("99.5%: {}".format(perc995))
        print("99.9%: {}".format(perc999))
        print("Max: {}".format(np.max(uncertainties)), '\n')

        plt.axvline(x=mean_ + std_, color='red', linestyle='dashed', label=r'$\mu+\sigma$: {:.3f}'.format(mean_ + std_))
        plt.axvline(x=mean_ + 2*std_, color='green', linestyle='dashed', label=r'$\mu+2\sigma$: {:.3f}'.format(mean_ + 2*std_))
        plt.axvline(x=perc95, color='black', linestyle='dashed', label='95%: {:.3f}'.format(perc95))
        plt.axvline(x=perc975, color='yellow', linestyle='dashed', label='97.5%: {:.3f}'.format(perc975))
        plt.axvline(x=perc99, color='magenta', linestyle='dashed', label='99%: {:.3f}'.format(perc99))
        plt.axvline(x=perc995, color='brown', linestyle='dashed', label='99.5%: {:.3f}'.format(perc995))
        plt.axvline(x=perc999, color='orange', linestyle='dashed', label='99.9%: {:.3f}'.format(perc999))

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
            "path": "./logs/train_agent_20230715_211722_rpf_v14/data.hdf5",
            "csv": "./logs/train_agent_20230715_211722_rpf_v14/rerun_test_scenarios_NU_uncerts.csv",
            "custom_marks": [],
            "bins": 100,
            "range": (0, 0.035),
        },
        {
            "name": "DAE DQN",
            "path": "./logs/train_agent_20231006_154948_dae_v5/data.hdf5",
            "csv": "./logs/train_agent_20231006_154948_dae_v5/rerun_test_scenarios_NU_uncerts.csv",
            "custom_marks": [],
            "bins": 100,
            "range": (-50, 100),
        },
        # {
        #     "name": "DAE DQN",
        #     "path": "./logs/train_agent_20230828_020015_ae_v22/data.hdf5",
        #     "csv": "./logs/train_agent_20230828_020015_ae_v22/rerun_test_scenarios_NU_uncerts.csv",
        #     "custom_marks": [],
        #     "bins": 100,
        #     "range": (-700, 0),
        # }
    ]

    plot_distributions(models, mode="csv")