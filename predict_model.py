#%%
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.datasets import Dataset, Two_Sample
from utils.model import ESN, ESNModel, ESNModel_DS, RCN, RCNModel,  RCNModel_DS, progress
import utils.measures as meas
import utils.dynamical_systems as ds

dynamical_system_name = 'lorenz'

if dynamical_system_name == 'lorenz':
    from lorenz.config import config

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

RC_type = config["MODEL"]["RC_type"]
tag = config["FILE_NAME_TAG"]

if RC_type == 'ESN':
    Network = ESN
    Model = ESNModel
    Model_DS = ESNModel_DS
elif RC_type == 'RCN':
    Network = RCN
    Model = RCNModel
    Model_DS = RCNModel_DS
else:
    print('RC not supported')

#%% load data

dataset_train = Dataset(
    num_trajectories = config["DATA"]["n_train"],
    len_trajectories = config["DATA"]["l_trajectories"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = True, 
    data_set_name = 'train',
    normalize_data = config["DATA"]["normalize_data"]
)
shift, scale = dataset_train.shift, dataset_train.scale
dataset_train.save_data()

dataset_test = Dataset(
    num_trajectories = config["DATA"]["n_test"],
    len_trajectories = config["DATA"]["l_trajectories_test"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'test',
    normalize_data = config["DATA"]["normalize_data"],
    shift = shift,
    scale = scale
)
dataset_test.save_data()

#%% load model
network = Network(
     config["MODEL"]["input_size"],
    config["MODEL"]["reservoir_size"],
    config["MODEL"]["hidden_size"],
    config["MODEL"]["input_size"],
    config["MODEL"]["scale_rec"],
    config["MODEL"]["scale_in"],
    config["MODEL"]["leaking_rate"],
)

model = Model(
    dataloader_train = None,
    dataloader_val = None,
    network = network,
)

model_name = config["PATH"] + tag + "_model_"
model.load_network(model_name)

#%%
load_samples = config["DATA"]["load_samples"]
load_sample_dists = config["DATA"]["load_sample_dists"]

#%% predict
warmup = config["DATA"]["max_warmup"]

if not load_samples:
    predictions, _ = model.integrate(
        torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
        T=dataset_test.input_data.shape[1] - warmup,
    )

#%%
folder = dynamical_system_name + "/predict"
    
if load_samples:
    nu1_trajs_true = None
    nu1_trajs_pred = None
    nu2_trajs_true = None
    nu2_trajs_pred = None
else:
    batch, T, d = predictions.shape
    true_trajs = dataset_test.output_data
    pred_trajs = predictions.detach().cpu().numpy()

    nu1_indices = []
    nu2_indices = []

    # separate trajectories based on their first coordinate when warmup ends
    for i in range(batch):
        z = true_trajs[i,warmup-1,:]
        if z[0] >=0:
            nu1_indices.append(i)
        else:
            nu2_indices.append(i)

    nu1_trajs_true = true_trajs[nu1_indices, :, :]
    nu1_trajs_pred = pred_trajs[nu1_indices, :, :]
    nu2_trajs_true = true_trajs[nu2_indices, :, :]
    nu2_trajs_pred = pred_trajs[nu2_indices, :, :]

    # save trajectories
    os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist
    np.save(os.path.join(folder, "nu1_trajs_true"+ tag), nu1_trajs_true)
    np.save(os.path.join(folder, "nu1_trajs_pred"+ tag), nu1_trajs_pred)
    np.save(os.path.join(folder, "nu2_trajs_true"+ tag), nu2_trajs_true)
    np.save(os.path.join(folder, "nu2_trajs_pred"+ tag), nu2_trajs_pred)


#%%
if not load_samples:
    num_bins = 100
    nu1 = nu1_trajs_true[:,warmup-1,:]
    nu2 = nu2_trajs_true[:,warmup-1,:]

    meas.plot_measure(nu2, (0,2), num_bins, 'hist')
#%%
print(folder)
#%% calculate distance between distributions for trajectories
name = "dist_trajs_truetrue_12" + tag + "_model_"
name1 = "nu1_trajs_true" + tag + "_model_"
name2 = "nu2_trajs_true" + tag + "_model_"
samples_12 = Two_Sample(nu1_trajs_true,
                        nu2_trajs_true, 
                        load_samples,
                        load_sample_dists,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists:
    sigma_kernel = samples_12.median_dist(100)
    print(f"median distance between points (averaged over time) is {sigma_kernel}")
    dists_12 = samples_12.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    

#%% calculate distance between distributions for trajectories
name = "dist_trajs_truepred_11" + tag + "_model_"
name1 = "nu1_trajs_true" + tag + "_model_"
name2 = "nu1_trajs_pred" + tag+ "_model_"
samples_11 = Two_Sample(nu1_trajs_true,
                        nu1_trajs_pred, 
                        load_samples,
                        load_sample_dists,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists:
    dists_11 = samples_11.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    
#%%
import importlib
importlib.reload(meas)
#%% plot the two distributions against each other

m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[:-1]
hline1 = meas.two_sample_test(m, alpha = 0.05, H0 = '==', biased = True)
hline2 = meas.two_sample_test(m, alpha = 0.05, H0 = '>eps', epsilon_sq = epsilon_sq, biased = True)

plt.figure(figsize=(8, 6))

# Plot time series
plt.plot(time, dists_11, label="MMD between $\mu_1$ and $\mu_2$ transported under Lorenz")
plt.plot(time, dists_12, label="MMD between $\mu_1$ transported under Lorenz and proxy")

plt.axvline(x=20, color="black", linestyle="--")
plt.axhline(y=hline1, linestyle=":", label=f"Crit val $H_0: \mu_1 = \mu_2$") 
plt.axhline(y=hline2, linestyle=":", label=f"Crit val $H_0: MMD(\mu_1, \mu_2)^2>{epsilon_sq}$")

plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("MMD")

plt.xlim(0, time[-1]) 
plt.ylim(0, 1e0)

plt.legend()
plt.savefig('MMD transport figure')
plt.show()

###############################################################################
###############################################################################
###############################################################################

# plot mu1, mu2 at time t=1000 and t=end, also under proxy
# plot stationary distribution of proxy and true
# make a function to make the sample smaller
# %%
