#%%
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.datasets import Dataset, Two_Sample, downsample_array
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
step = config["DATA"]["step"]

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
    shift = shift, # use same shift and scale from training data
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
T_end = dataset_test.input_data.shape[1]

if not load_samples:
    predictions, _ = model.integrate(
        torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
        T=T_end - warmup,
    )

#%%
folder = dynamical_system_name + "/predict"
    
if load_samples:
    nu1_trajs_true = np.load(os.path.join(folder, "nu1_trajs_true"+ tag+'.npy'))
    nu1_trajs_pred = np.load(os.path.join(folder, "nu1_trajs_pred"+ tag+'.npy'))
    nu2_trajs_true = np.load(os.path.join(folder, "nu2_trajs_true"+ tag+'.npy'))
    nu2_trajs_pred = np.load(os.path.join(folder, "nu2_trajs_pred"+ tag+'.npy'))
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

    # ensure that samples are of equal size
    n_sample_new = 1000
    nu1_indices = downsample_array(np.array(nu1_indices), n_sample_new)
    nu2_indices = downsample_array(np.array(nu2_indices), n_sample_new)

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
i = 0 #np.random.randint(1000)

true_traj = nu1_trajs_true[i]
pred_traj = nu1_trajs_pred[i]
time_steps = dataset_test.tt[:-1]

coords = ['x', 'y', 'z']
warmup_time = warmup * step
t_end = T_end * step
x_lim = (0,t_end)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(time_steps, true_traj[:, i], label="True")
    ax.plot(time_steps, pred_traj[:, i], label="Predicted")
    ax.axvline(warmup_time, linestyle=":", color = "black", label="Warmup" if i == 0 else None)

    ax.set_xlim(x_lim)
    ax.set_ylabel(coords[i])
    if i == 0:
        ax.legend(loc="upper right")

axes[-1].set_xlabel("Time")
# fig.suptitle("Prediction for one trajectory")
fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

figures_folder = folder + '/fig'
os.makedirs(figures_folder, exist_ok=True)
fig_path = os.path.join(figures_folder, 'Prediction_one_trajectory.png')
plt.savefig(fig_path, dpi=300)
plt.show()
plt.close(fig)

#%% check distribution of one of the measures
if not load_samples:
    num_bins = 100
    nu1 = nu1_trajs_true[:,warmup-1,:]
    nu2 = nu2_trajs_true[:,warmup-1,:]

    meas.plot_measure(nu2, (0,2), num_bins, 'hist')

#%% calculate distance between distributions for trajectories
name = "dist_trajs_truetrue_12" + tag + "_model_"
name1 = "nu1_trajs_true" + tag
name2 = "nu2_trajs_true" + tag
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
else:
    dists_12 = samples_12.dist
    

#%% calculate distance between distributions for trajectories
name = "dist_trajs_truepred_11" + tag + "_model_"
name1 = "nu1_trajs_true" + tag
name2 = "nu1_trajs_pred" + tag
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
else:
    dists_11 = samples_11.dist
    
# #%%
# import importlib
# importlib.reload(meas)
#%% plot the two distributions against each other

m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[:-1]
hline1 = meas.two_sample_test(m, alpha = 0.05, H0 = '==', biased = True)
hline2 = meas.two_sample_test(m, alpha = 0.05, H0 = '>eps', epsilon_sq = epsilon_sq, biased = True)

print(f"test == crit val{hline1}")
print(f"test >eps crit val{hline2}")
plt.figure(figsize=(8, 6))

# Plot time series
plt.plot(time, dists_12, label="MMD$^2$ between $\mu_a$ and $\mu_b$ transported under Lorenz")
plt.plot(time, dists_11, label="MMD$^2$ between $\mu_a$ transported under Lorenz and proxy")

plt.axvline(x=warmup * step, color="black", linestyle="--")
plt.axhline(y=hline1, linestyle=":", label=f"Crit val $H_0: \mu_a = \mu_b$") 
plt.axhline(y=hline2, linestyle=":", label=f"Crit val $H_0: MMD(\mu_a, \mu_b)^2>{epsilon_sq}$", color='red')

plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("MMD$^2$")

plt.xlim(0, 80) 
plt.ylim(0, 1e0)

plt.legend()
fig_path = os.path.join(figures_folder, 'MMD transport figure.png')
plt.savefig(fig_path, dpi=300)
plt.show()

#%% plot distributions at warmup and end
import matplotlib.gridspec as gridspec

indices_plot = [0,2]
mu1 = 100 * samples_12.mu1[: , warmup-1, indices_plot]
mu2 = 100 * samples_12.mu2[: , warmup-1, indices_plot]
mu3 = 100 * samples_12.mu1[: , -1, indices_plot]
mu4 = 100 * samples_12.mu2[: , -1, indices_plot]

datasets = [mu1, mu2, mu3, mu4]
titles   = [f"$\mu_1$ at t={warmup * step}", f"$\mu_2$ at t={warmup * step}", f"$\mu_1$ at t={T_end * step}", f"$\mu_2$ at t={T_end * step}"]

xlim = (-20, 20)
ylim = (0, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]

for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="viridis")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts")

fig.suptitle("Densities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_truetrue.png')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%%

indices_plot = [0,2]
mu1 = 100 * samples_11.mu1[: , warmup-1, indices_plot] # rescale to lorenz scale
mu2 = 100 * samples_11.mu2[: , warmup-1, indices_plot]
mu3 = 100 * samples_11.mu1[: , -1, indices_plot]
mu4 = 100 * samples_11.mu2[: , -1, indices_plot]

datasets = [mu1, mu2, mu3, mu4]
titles   = [f"$\mu_t$ at t={warmup * step}", f"$\mu_p$ at t={warmup * step}", f"$\mu_t$ at t={T_end * step}", f"$\mu_p$ at t={T_end * step}"]

xlim = (-20, 20)
ylim = (0, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]

for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="viridis")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts")

fig.suptitle("Densities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_truepred.png')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
indices_plot = [0,2]
mu1 = 100 * samples_12.mu1[: , warmup-1, indices_plot]
mu2 = 100 * samples_12.mu2[: , warmup-1, indices_plot]
mu3 = 100 * samples_11.mu2[:, warmup-1, indices_plot]
mu4 = 100 * samples_12.mu1[: , -1, indices_plot]
mu5 = 100 * samples_12.mu2[: , -1, indices_plot]
mu6 = 100 * samples_11.mu2[:, -1, indices_plot]

datasets = [mu2, mu1, mu3, mu5, mu4, mu6]
titles   = [f"$\mu^2_t$ at t={warmup * step}", f"$\mu^1_t$ at t={warmup * step}", r"$\hat{\mu}^1_t$ at " + f"t={T_end * step}", f"$\mu^2_t$ at t={T_end * step}", f"$\mu^1_t$ at t={T_end * step}", r"$\hat{\mu}^1_t$ at" + f" t={T_end * step}"]

xlim = (-20, 20)
ylim = (0, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])]

for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="viridis")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts")


fig.text(0.16, 0.97, "(a)", ha="center", va="center", fontsize=14)
fig.text(0.45, 0.97, "(b)", ha="center", va="center", fontsize=14)
fig.text(0.74, 0.97, "(c)", ha="center", va="center", fontsize=14)

# fig.suptitle("Densities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_truetruepred.png')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
#%% compare the invariant measures of lorenz and ESN
# find invariant measure lorenz

n_init_cond_meas = 1000
t_start = warmup 
t_end = 1500
z0 = np.zeros((3))
sd_meas = 300

find_invar_meas = True
if find_invar_meas:
    lor = ds.lorenz()
    mu = meas.invariant_measure(lor,n_init_cond_meas, t_start, t_end, z0, sd_meas )
    np.save(folder + '/Lorenz invariant measure', mu)
else:
    mu = np.load(folder + '/Lorenz invariant measure.npy')

#%% invariant measure ESN
N = network.reservoir_size
x0 = np.zeros((N))
sd_meas = 300
model_ds = Model_DS(model)

find_invar_meas = True
if find_invar_meas:
    zeta_ESN = meas.invariant_measure(model_ds, n_init_cond_meas, t_start, t_end, x0, sd_meas )
    zeta_ESN = torch.from_numpy(zeta_ESN)
    mu_ESN = network.readout(zeta_ESN)
    mu_ESN = mu_ESN.detach().cpu().numpy()
    # np.save('Lorenz ESN invariant measure state space', zeta_ESN) too large
    np.save(folder + '/Lorenz ESN invariant measure readout', mu_ESN)
else:
    # zeta_ESN = np.load('Lorenz ESN invariant measure state space.npy')
    mu_ESN = np.load(folder + '/Lorenz ESN invariant measure readout.npy')

#%% plot invariant measures
from matplotlib.colors import LogNorm
indices_plot = [0,2]
mu1 = mu[:, indices_plot]
mu2 = 100 * mu_ESN[:, indices_plot] # rescale to actual scale

datasets = [mu1, mu2]
titles   = ["Lorenz", "Proxy"]

xlim = (-20, 20)
ylim = (-45, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]
for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim,ylim], cmap="viridis", norm = LogNorm(clip=True))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts (log scale)")

fig.suptitle("Invariant measures", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'invariant_measures.png')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%% calculate distance between trajectories for one set of trajectories

# true and predicted first trajectory
traj_1, traj_2 = samples_11.mu1[0], samples_11.mu2[0]
diff_traj = traj_1 - traj_2
dist_trajs_truepred = np.linalg.norm(diff_traj, axis=1)

# comparing two trajectories under the true system
traj_1, traj_2 = samples_12.mu1[0], samples_12.mu2[0]
diff_traj = traj_1 - traj_2
dist_trajs_truetrue = np.linalg.norm(diff_traj, axis=1)

time = dataset_test.tt[:-1]

plt.figure(figsize=(8, 5))

plt.plot(time, dist_trajs_truetrue, label="distance between $m_a$ and $m_b$ transported under Lorenz")
plt.plot(time, dist_trajs_truepred, label="distance between $m_a$ transported under Lorenz and proxy")

plt.yscale("log")
plt.xlim(0, 80)       # example x range
plt.ylim(0, 1)     # example y range; adjust to your data

plt.xlabel("Time")
plt.ylabel("Distance")
# plt.title("Trajectory Distance Comparison", fontsize=14)

plt.axvline(x=step * warmup, color="black", linestyle="--")

plt.legend()
plt.tight_layout()

fig_path = os.path.join(figures_folder, 'Trajectories distance figure.png')
plt.savefig(fig_path, dpi=300)
plt.show()


#%%
#%% calculate distance between trajectories average over all trajectories

# true and predicted first trajectory
traj_1, traj_2 = samples_11.mu1, samples_11.mu2
diff_traj = traj_1 - traj_2
dist_trajs_truepred = np.mean(np.linalg.norm(diff_traj, axis=2), axis=0)

# comparing two trajectories under the true system
traj_1, traj_2 = samples_12.mu1, samples_12.mu2
diff_traj = traj_1 - traj_2
dist_trajs_truetrue = np.mean(np.linalg.norm(diff_traj, axis=2), axis=0)

time = dataset_test.tt[:-1]

plt.figure(figsize=(8, 5))

plt.plot(time, dist_trajs_truetrue, label="mean distance between $m_a$ and $m_b$ transported under Lorenz")
plt.plot(time, dist_trajs_truepred, label="mean distance between $m_a$ transported under Lorenz and proxy")

plt.yscale("log")
plt.xlim(0, 80)       # example x range
plt.ylim(0, 1)     # example y range; adjust to your data

plt.xlabel("Time")
plt.ylabel("Distance")
# plt.title("Trajectory Distance Comparison", fontsize=14)

plt.axvline(x=step * warmup, color="black", linestyle="--")

plt.legend()
plt.tight_layout()

fig_path = os.path.join(figures_folder, 'Trajectories mean distance figure.png')
plt.savefig(fig_path, dpi=300)
plt.show()



###############################################################################
###############################################################################
###############################################################################

# plot mu1, mu2 at time t=1000 and t=end, also under proxy
# plot stationary distribution of proxy and true
# plot true and proxy for one trajectory
# make a function to make the sample smaller

# 29 Sep 2025
# check that the true and predicted trajectories are the same ones. They don't look the same above.
# also, why is the warmup so bad?
# figure out why invariant measures aren't there
# detach in integration of model, could make things faster, no need for grad in tensor

# %%

samples_11.dist[1000]

# %%

np.max(samples_12.mu1[:,:1000]-samples_12.mu2[:,:1000])

# %%
