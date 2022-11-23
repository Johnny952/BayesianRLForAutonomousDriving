import h5py

if __name__ == "__main__":
    path = './logs/train_agent_20221114_203403/data.hdf5'

    with h5py.File(path, "r") as f:
        print(f.keys())
        for key in f.keys():
            print(f"Keys {key}:", f[key].keys())
            for key_ in f[key].keys():
                print(f"Keys {key}-{key_}:", f[key][key_].keys())
            print('')

        print(f["5"]["5"]["collision"][()])