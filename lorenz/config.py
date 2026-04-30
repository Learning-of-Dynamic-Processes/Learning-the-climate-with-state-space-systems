import torch
import numpy as np

config = {}


config["MODEL"] = {}
config["TRAINING"] = {}
config["DATA"] = {}

config["MODEL"]["RC_type"] = "ESN"
config["TRAINING"]["ridge"] = True

config["MODEL"]["input_size"] = 3

if config["TRAINING"]["ridge"]:
    config["MODEL"]["hidden_size"] = []
else:
    config["MODEL"]["hidden_size"] = [1000, 1000] #  [64, 64]

if config["TRAINING"]["ridge"]:
    if config["MODEL"]["RC_type"] == "ESN":
        config["MODEL"]["reservoir_size"] = 1000
        config["MODEL"]["scale_rec"] = 1.2
        config["MODEL"]["scale_in"] = 13/3
        config["MODEL"]["leaking_rate"] = 1

    elif config["MODEL"]["RC_type"] == "RCN":
        config["MODEL"]["reservoir_size"] = 1000
        config["MODEL"]["scale_rec"] = 1.2
        config["MODEL"]["scale_in"] = 13/3
        config["MODEL"]["leaking_rate"] = 1

        config["MODEL"]["scale_rec_list"] = [0.5, 0.6, 0.7, 0.8, 1.5] # [0.7, 0.9, 0.99, 1, 1.2, 1.5, 2, 3] # [0.7, 0.8, 0.9, 0.95, 0.99, 1, 1.1, 1.2, 1.5, 2, 2.5, 3]
        config["MODEL"]["scale_in_list"] = [10, 11, 12, 13, 14, 15, 16] # [13, 16, 20, 23, 26] # [0.5, 1, 2, 5, 7, 10, 13, 17, 20, 25, 30]
        config["MODEL"]["leaking_rate_list"] = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1] # [0.5, 0.75, 0.9, 0.95, 0.99, 1]  # [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
else:
    if config["MODEL"]["RC_type"] == "ESN":
        config["MODEL"]["reservoir_size"] = 1000
        config["MODEL"]["scale_rec"] = 1.2
        config["MODEL"]["scale_in"] = 13/3
        config["MODEL"]["leaking_rate"] = 1

    elif config["MODEL"]["RC_type"] == "RCN":
        config["MODEL"]["reservoir_size"] = 2**11
        config["MODEL"]["scale_rec"] = 1.2
        config["MODEL"]["scale_in"] = 13/3
        config["MODEL"]["leaking_rate"] = 1

config["TRAINING"]["epochs"] = 100
config["TRAINING"]["batch_size"] = 20
config["TRAINING"]["learning_rate"] = 5e-3
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100
config["TRAINING"]["offset"] = 1000
config["TRAINING"]["device"] = "cpu"
config["TRAINING"]["ridge_factor"] = 3.1622776601683794e-15 # 1e-9

config["DATA"]["dynamical_system_name"] = 'lorenz'
config["DATA"]["parameters"] = (10, 8/3, 28)
config["DATA"]["max_warmup"] = 1000
config["DATA"]["step"] = 0.02
config["DATA"]["y0"] = np.array([0,0,27]).astype(np.float64)
config["DATA"]["initial_points_sd"] = 20
config["DATA"]["n_train"] = 100
config["DATA"]["n_val"] = 30
config["DATA"]["n_test"] = 2200
config["DATA"]["l_trajectories"] = 3000
config["DATA"]["l_trajectories_test"] = 4000
config["DATA"]["data_type"] = torch.float64
config["DATA"]["method"] = 'RK4'
config["DATA"]["load_data"] = True
config["DATA"]["load_samples"] = True
config["DATA"]["load_sample_dists_mmd"] = True
config["DATA"]["load_sample_dists_wass1"] = True
config["DATA"]["load_sample_dists_wass1_traj"] = True
config["DATA"]["normalize_data"] = True
config["PATH"] = "lorenz/models/"
config["FILE_NAME_TAG"] = config["MODEL"]["RC_type"] + '_ridge_' + str(config["TRAINING"]["ridge"])



# config["MODEL"]["leaking_rate"] = 0.5012724887553683
# config["TRAINING"]["ridge_factor"] = 1e-3
# config["MODEL"]["scale_rec"] = 0.8048300803863692
# config["MODEL"]["scale_in"] = 0.17928881689644652