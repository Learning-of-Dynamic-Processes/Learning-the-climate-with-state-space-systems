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
load_sample_dists_mmd = config["DATA"]["load_sample_dists_mmd"]
load_sample_dists_wass1 = config["DATA"]["load_sample_dists_wass1"]
load_sample_dists_wass1_traj = config["DATA"]["load_sample_dists_wass1_traj"]
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

#%% check prediction for one trajectory
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
fig_path = os.path.join(figures_folder, 'Prediction_one_trajectory.pdf')
plt.savefig(fig_path, dpi=300)
plt.show()
plt.close(fig)

#%% check distribution of one of the measures
if not load_samples:
    num_bins = 100
    nu1 = nu1_trajs_true[:,warmup-1,:]
    nu2 = nu2_trajs_true[:,warmup-1,:]

    meas.plot_measure(nu2, (0,2), num_bins, 'hist')

#%% calculate distance between distributions for trajectories samples_12_true
name = "dist_trajs_truetrue_12" + tag + "_model_"
name1 = "nu1_trajs_true" + tag
name2 = "nu2_trajs_true" + tag
samples_12 = Two_Sample(nu1_trajs_true,
                        nu2_trajs_true, 
                        load_samples,
                        load_sample_dists_mmd,
                        load_sample_dists_wass1,
                        load_sample_dists_wass1_traj,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists_mmd:
    sigma_kernel = samples_12.median_dist(100)
    print(f"median distance between points (averaged over time) is {sigma_kernel}")
    dists_12_mmd = samples_12.calculate_dist_mmd(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
else:
    dists_12_mmd = samples_12.dist_mmd
    
if not load_sample_dists_wass1:
    dists_12_wass1 = samples_12.calculate_dists_wass1(None)
else:
    dists_12_wass1 = samples_12.dists_wass1


aggregate = 10
if not load_sample_dists_wass1_traj:
    coord_list = None
    init_cond = 0
    dists_12_wass1_traj = samples_12.calculate_dists_wass1_traj(coord_list, init_cond, warmup, aggregate)
else:
    dists_12_wass1_traj = samples_12.dists_wass1_traj

#%% calculate distance between distributions for trajectories samples_1_truepred
name = "dist_trajs_truepred_11" + tag + "_model_"
name1 = "nu1_trajs_true" + tag
name2 = "nu1_trajs_pred" + tag
samples_11 = Two_Sample(nu1_trajs_true,
                        nu1_trajs_pred, 
                        load_samples,
                        load_sample_dists_mmd,
                        load_sample_dists_wass1,
                        load_sample_dists_wass1_traj,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists_mmd:
    dists_11_mmd = samples_11.calculate_dist_mmd(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
else:
    dists_11_mmd = samples_11.dist_mmd

    
if not load_sample_dists_wass1:
    dists_11_wass1 = samples_11.calculate_dists_wass1()
else:
    dists_11_wass1 = samples_11.dists_wass1

if not load_sample_dists_wass1_traj:
    coord_list = None
    init_cond = 0
    aggregate = 10
    dists_11_wass1_traj = samples_11.calculate_dists_wass1_traj(coord_list, init_cond, warmup, aggregate)
else:
    dists_11_wass1_traj = samples_11.dists_wass1_traj

#%% plot the MMD between the distributions against each other

m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[:-1]
hline1 = meas.two_sample_test(m, alpha = 0.05, H0 = '==', biased = True)
hline2 = meas.two_sample_test(m, alpha = 0.05, H0 = '>eps', epsilon_sq = epsilon_sq, biased = True)

print(f"test == crit val{hline1}")
print(f"test >eps crit val{hline2}")
plt.figure(figsize=(8, 5))

# Plot time series
plt.plot(time, dists_12_mmd, label=r"MMD$^2$ between $\mu^1_{\tau}$ and $\mu^2_{\tau}$")
plt.plot(time, dists_11_mmd, label=r"MMD$^2$ between $\mu^1_{\tau}$ and $\hat{\mu}^1_{\tau}$")

plt.axvline(x=warmup * step, color="black", linestyle="--")
plt.axhline(y=hline1, linestyle=":", label=f"Crit val $H_0: \mu^a = \mu^b$") 
plt.axhline(y=hline2, linestyle=":", label=f"Crit val $H_0: MMD(\mu^a, \mu^b)^2>{epsilon_sq}$", color='red')

plt.yscale("log")
plt.xlabel(r"Time $\tau$")
plt.ylabel("MMD$^2$")

plt.xlim(0, 80) 
plt.ylim(0, 1e0)
plt.tight_layout()

plt.legend()
fig_path = os.path.join(figures_folder, 'MMD transport figure.pdf')
plt.savefig(fig_path, dpi=300)
plt.show()

#%% plot the wass_dist between the distributions against each other

m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[:-1]
coords = ['x','y','z']
# hline1 = meas.two_sample_test(m, alpha = 0.05, H0 = '==', biased = True)
# hline2 = meas.two_sample_test(m, alpha = 0.05, H0 = '>eps', epsilon_sq = epsilon_sq, biased = True)

# print(f"test == crit val{hline1}")
# print(f"test >eps crit val{hline2}")
plt.figure(figsize=(8, 5))

# Plot time series
for i,c in zip(range(dists_12_wass1.shape[0]), coords):
    plt.plot(time, dists_12_wass1[i], label=c + r" axis W$_1$ between $\mu^1_{\tau}$ and $\mu^2_{\tau}$")
    plt.plot(time, dists_11_wass1[i], label=c + r" axis W$_1$ between $\mu^1_{\tau}$ and $\hat{\mu}^1_{\tau}$")

plt.axvline(x=warmup * step, color="black", linestyle="--")
# plt.axhline(y=hline1, linestyle=":", label=f"Crit val $H_0: \mu^a = \mu^b$") 
# plt.axhline(y=hline2, linestyle=":", label=f"Crit val $H_0: MMD(\mu^a, \mu^b)^2>{epsilon_sq}$", color='red')

plt.yscale("log")
plt.xlabel(r"Time $\tau$")
plt.ylabel("W$_1$")

plt.xlim(0, 80) 
plt.ylim(0.8 * 1e-3, 1.5 * 1e-1)
plt.tight_layout()

plt.legend(loc = "upper right")
fig_path = os.path.join(figures_folder, 'Wass1 transport figure.pdf')
plt.savefig(fig_path, dpi=300)
plt.show()

#%% plot wasserstein distances in panel

m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[:-1]
coords = ['x', 'y', 'z']
labels_graphs = ['(a) x-axis', '(b) y-axis', '(c) z-axis']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (c, ax) in enumerate(zip(coords, axes)):
    line1, = ax.plot(time, dists_12_wass1[i], label=c + r" axis W$_1$ between $\mu^1_{\tau}$ and $\mu^2_{\tau}$")
    line2, = ax.plot(time, dists_11_wass1[i], label=c + r" axis W$_1$ between $\mu^1_{\tau}$ and $\hat{\mu}^1_{\tau}$")
    ax.axvline(x=warmup * step, color="black", linestyle="--")
    ax.set_yscale("log")
    if i == 1:
        ax.set_xlabel(r"Time $\tau$")
    ax.set_title(labels_graphs[i])
    if i == 0:
        ax.set_ylabel("W$_1$")
    ax.set_xlim(0, 80)
    ax.set_ylim(0.8 * 1e-3, 1.5 * 1e-1)
    # ax.legend(loc="upper right")

fig.legend(
    [line1, line2],
    [r"W$_1(\mu^1_{\tau}, \mu^2_{\tau})$", r"W$_1(\mu^1_{\tau}, \hat{\mu}^1_{\tau})$"],
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False
)


plt.tight_layout()
fig_path = os.path.join(figures_folder, 'Wass1 transport figure_panel.pdf')
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
titles   = [r"$\mu^1$ at $\tau$" + f"={warmup * step}", r"$\mu^2$ at $\tau$" + f"={warmup * step}", r"$\mu^1$ at $\tau$" + f"={T_end * step}", r"$\mu^2$ at $\tau$" + f"={T_end * step}"]

xlim = (-20, 20)
ylim = (0, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]

for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="magma_r")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts")

# fig.suptitle("Densities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_truetrue.pdf')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%%

indices_plot = [0,2]
mu1 = 100 * samples_11.mu1[: , warmup-1, indices_plot] # rescale to lorenz scale
mu2 = 100 * samples_11.mu2[: , warmup-1, indices_plot]
mu3 = 100 * samples_11.mu1[: , -1, indices_plot]
mu4 = 100 * samples_11.mu2[: , -1, indices_plot]

datasets = [mu1, mu2, mu3, mu4]
titles   = [r"$\mu^1$ at $\tau$" + f"= {warmup * step}", r"$\hat{\mu}^1$ at $\tau$" + f"={warmup * step}", r"$\mu^1$ at $\tau$" + f"={T_end * step}", r"$\hat{\mu}^1$ at $\tau$" + f"={T_end * step}"]

xlim = (-20, 20)
ylim = (0, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]

for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="magma_r")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts")

# fig.suptitle("Densities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_truepred.pdf')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%%
import matplotlib.gridspec as gridspec

indices_plot = [0, 2]
xlim = (-20, 20)
ylim = (0, 50)

# --- Data ---
# Column 1: mu^2
col1 = [100 * samples_12.mu2[:, warmup-1, indices_plot],
        100 * samples_12.mu2[:, -1,       indices_plot]]

# Column 2: mu^1
col2 = [100 * samples_12.mu1[:, warmup-1, indices_plot],
        100 * samples_12.mu1[:, -1,       indices_plot]]

# Column 3: hat{mu}^1
col3 = [100 * samples_11.mu2[:, warmup-1, indices_plot],
        100 * samples_11.mu2[:, -1,       indices_plot]]

columns   = [col1, col2, col3]
col_titles = [r"(a) $\mu^2$", r"(b) $\mu^1$", r"(c) $\hat{\mu}^1$"]
row_labels = [r"$\tau$" + f"$= {warmup * step}$", r"$\tau$" + f"$= {T_end * step}$"]

# --- Figure ---
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.2, wspace=0.3)

axes = [[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(2)]

h_last = None
for col_idx, (col_data, col_title) in enumerate(zip(columns, col_titles)):
    for row_idx, data in enumerate(col_data):
        ax = axes[row_idx][col_idx]
        x, y = data[:, 0], data[:, 1]
        h = ax.hist2d(x, y, bins=100, range=[xlim, ylim], cmap="magma_r")
        h_last = h

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if row_idx == 1:
            ax.set_xlabel("x")

        # Column titles on top row only
        if row_idx == 0:
            ax.set_title(col_title, fontsize=13)

        # Row labels on leftmost column only
        if col_idx == 0:
            ax.set_ylabel(f"{row_labels[row_idx]}\nz", fontsize=10)
        # else:
        #     ax.set_ylabel("z")

# Shared colorbar
cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(h_last[3], cax=cax)
cbar.set_label("Counts")

fig_path = os.path.join(figures_folder, 'densities_combined.pdf')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%% plot initial and final distributions for true and predicted
from scipy.stats import gaussian_kde

row1 = [samples_12.mu2[:, warmup-1, :], samples_11.mu1[:, warmup-1, :], samples_11.mu2[:, warmup-1, :]]
row2 = [samples_12.mu2[:, -1, :], samples_11.mu1[:, -1, :], samples_11.mu2[:, -1, :]]
d = row1[0].shape[1]
n_datasets = len(row1)
bins = 100
kde = True

coords = ['x', 'y', 'z']
row_labels = [r'(a) Initial distribution $\tau=20$', r'(b) Final distribution $\tau = 80$']
data_set_labels = [r'True $\mu^2_\tau$', r'True $\mu^1_\tau$', r'Predicted $\hat{\mu}^1_\tau$']

fig, axes = plt.subplots(2, d, figsize=(5 * d, 8))

for row_idx, row in enumerate([row1, row2]):
    for dim_idx in range(d):
        for ds_idx, data_set in enumerate(row):
            ax = axes[row_idx, dim_idx]
            data = data_set[:, dim_idx]
            
            if kde:
                grid = np.linspace(data.min(), data.max(), 300)
                ax.plot(grid, gaussian_kde(data)(grid), color=f"C{ds_idx}", label = data_set_labels[ds_idx] if (row_idx ==0 and dim_idx == 0) else "")
                ax.fill_between(grid, gaussian_kde(data)(grid), alpha=0.3, color=f"C{ds_idx}")
            else:
                ax.hist(data, bins=bins, alpha=0.7, density=True, color=f"C{ds_idx}")

        ax.set_xlabel(coords[dim_idx] if row_idx == 1 else "")
        ax.set_ylabel(row_labels[row_idx] if dim_idx == 0 else "")
            
        # ax.set_title(f"{row_labels[ds_idx]} — {coords[dim_idx]}")

handles = [axes[0, 0].get_legend_handles_labels()[0][i] for i in range(len(data_set_labels))]


fig.legend(
    handles, data_set_labels,
    loc='lower center',
    ncol=n_datasets,
    bbox_to_anchor=(0.5, -0.05),  # just below the figure
    frameon=False
)

plt.tight_layout()

# Save the figure
fig_path = os.path.join(figures_folder, 'densities_axes.pdf')
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

#%% plot converged distribution for one trajectory
cutoff = aggregate
n, T, d = samples_11.mu1.shape
invar_meas = samples_11.mu1[:,T-cutoff:, : ].reshape(-1,d)
row = [invar_meas, samples_11.mu2[0, warmup:, :]]
n_datasets = len(row)
bins = 100
kde = True

coords = ['x', 'y', 'z']
data_set_labels = ['True', 'Predicted']

fig, axes = plt.subplots(1, d, figsize=(5 * d, 8))

for dim_idx in range(d):
    for ds_idx, data_set in enumerate(row):
        ax = axes[dim_idx]
        data = data_set[:, dim_idx]
        
        if kde:
            grid = np.linspace(data.min(), data.max(), 300)
            ax.plot(grid, gaussian_kde(data)(grid), color=f"C{ds_idx}", label = data_set_labels[ds_idx] if dim_idx == 0 else None)
            ax.fill_between(grid, gaussian_kde(data)(grid), alpha=0.3, color=f"C{ds_idx}")
        else:
            ax.hist(data, bins=bins, alpha=0.7, density=True, color=f"C{ds_idx}")

    ax.set_xlabel(coords[dim_idx])
        

handles = [axes[0].get_legend_handles_labels()[0][i] for i in range(len(data_set_labels))]


fig.legend(
    handles, data_set_labels,
    loc='lower center',
    ncol=n_datasets,
    bbox_to_anchor=(0.5, -0.05),  # just below the figure
    frameon=False
)

plt.tight_layout()


#%% plot wasserstein distance for one trajectory to invariant measure over time
m = samples_12.mu1.shape[0]
print(f"sample size is {m}")
epsilon_sq = 0.1
time = dataset_test.tt[warmup:-1]
coords = ['(a) x', '(b) y', '(c) z']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (c, ax) in enumerate(zip(coords, axes)):
    ax.plot(time, samples_11.dists_wass1_traj[i], label=c + r" axis W$_1$ between $\mu$ and $\hat{\mu}^1_{\tau}$")
    # ax.axvline(x=warmup * step, color="black", linestyle="--")
    ax.set_yscale("log")
    if i == 1:
        ax.set_xlabel(f"Time $\\tau$\n"+coords[i]+"-axis")
    else:
        ax.set_xlabel(f" \n"+ coords[i]+"-axis")
    if i == 0:
        ax.set_ylabel("W$_1$ distance")
    ax.set_xlim(warmup_time, 80)
    # ax.set_ylim(0.8 * 1e-3, 1.5 * 1e-1)
    # ax.legend(loc="upper right")

plt.tight_layout()
fig_path = os.path.join(figures_folder, 'Wass1 transport single trajectory figure_panel.pdf')
plt.savefig(fig_path, dpi=300)
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
titles   = ["(a) Lorenz", "(b) Proxy"]

xlim = (-20, 30)
ylim = (-40, 50)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]
for i, (data, ax) in enumerate(zip(datasets, axes)):
    x = data[:, 0]
    y = data[:, 1]
    h = ax.hist2d(x, y, bins=100, range=[xlim,ylim], cmap="magma_r", norm = LogNorm(clip=True))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("z")
    ax.set_title(titles[i])

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label("Counts (log scale)")

# fig.suptitle("Invariant measures", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig_path = os.path.join(figures_folder, 'invariant_measures.pdf')
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

fig_path = os.path.join(figures_folder, 'Trajectories distance figure.pdf')
plt.savefig(fig_path, dpi=300)
plt.show()


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

plt.plot(time, dist_trajs_truetrue, label=r"mean distance between $\mu^1_{\tau}$ and $\mu^2_{\tau}$")
plt.plot(time, dist_trajs_truepred, label=r"mean distance between $\mu^1_{\tau}$ and $\hat{\mu}^1_{\tau}$")

plt.yscale("log")
plt.xlim(0, 80)       # example x range
plt.ylim(0, 1)     # example y range; adjust to your data

plt.xlabel(r"Time $\tau$")
plt.ylabel("Distance")
# plt.title("Trajectory Distance Comparison", fontsize=14)

plt.axvline(x=step * warmup, color="black", linestyle="--")

plt.legend()
plt.tight_layout()

fig_path = os.path.join(figures_folder, 'Trajectories mean distance figure.pdf')
plt.savefig(fig_path, dpi=300)
plt.show()

# %%
