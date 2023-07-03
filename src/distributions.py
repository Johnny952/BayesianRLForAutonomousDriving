import h5py
import numpy as np
import matplotlib.pyplot as plt

NB_BINS = 40

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

def plot_distributions(models):
    for model in models:
        _, uncertainties = read_file(model["path"])
        marks = model["custom_marks"]
        plt.figure()
        plt.hist(uncertainties, bins=NB_BINS)
        plt.xlabel("Uncertainty")
        model_name = model["name"]

        mean_ = np.mean(uncertainties)
        std_ = np.std(uncertainties)
        print(model_name)
        print(mean_ + std_)
        print(mean_ + 2*std_, '\n')


        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        plt.text(mean_ + std_ - max_xlim*0.05, min_ylim, r'$\mu+\sigma$', rotation=0)# - max_ylim*0.1
        plt.axvline(x=mean_ + std_, color='red', linestyle='dashed')

        plt.text(mean_ + 2*std_ - max_xlim*0.05, min_ylim, r'$\mu+2\sigma$', rotation=0)
        plt.axvline(x=mean_ + 2*std_, color='red', linestyle='dashed')

        for m in marks:
            plt.text(m - max_xlim*0.07, max_ylim*0.05, 'Custom', rotation=0)
            plt.axvline(x=m, color='black', linestyle='dashed')

        plt.title(model_name)
        plt.savefig(f"./videos/dist_{model_name}.png")
        plt.close()

if __name__ == "__main__":
    models = [
        {
            "name": "Ensemble RPF DQN",
            "path": "./logs/train_agent_20230628_172622_rpf_v10/data.hdf5",
            "custom_marks": []
        },
        {
            "name": "DAE DQN",
            "path": "./logs/train_agent_20230628_172734_ae_v10/data.hdf5",
            "custom_marks": [110]
        }
    ]

    plot_distributions(models)