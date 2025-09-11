#%%
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.datasets import Dataset
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

dataset_test = Dataset(
    None,
    None,
    None,
    load_data = True,
    data_set_name = 'test'
)
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

model_name = config["PATH"] + RC_type + "_ridge_" +  str(config["TRAINING"]["ridge"]) + "_model_"
model.load_network(model_name)

#%% predict
warmup = config["DATA"]["max_warmup"]
predictions, _ = model.integrate(
    torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
    T=dataset_test.input_data.shape[1] - warmup,
)

#%%

batch, T, d = predictions.shape
true_trajs = dataset_test.output_data
pred_trajs = predictions.detach()

nu1_indices = []
nu2_indices = []
for i in range(batch):
    z = true_trajs[i,warmup-1,:]
    if z[0] >=0:
        nu1_indices.append(i)
    else:
        nu2_indices.append(i)
n_nu1 = len(nu1_indices)
n_nu2 = len(nu2_indices)
nu1_trajs_true = np.zeros((n_nu1, T, d))
nu1_trajs_pred = np.zeros((n_nu1, T, d))
nu2_trajs_true = np.zeros((n_nu2, T, d))
nu2_trajs_pred = np.zeros((n_nu2, T, d))

for i in range(n_nu1):
    nu1_trajs_true[i,:,:] = true_trajs[nu1_indices[i], :, :]
    nu1_trajs_pred[i,:,:] = pred_trajs[nu1_indices[i], :, :]
for i in range(n_nu2):
    nu2_trajs_true[i,:,:] = true_trajs[nu2_indices[i], :, :]
    nu2_trajs_pred[i,:,:] = pred_trajs[nu2_indices[i], :, :]

#%%
num_bins = 100
nu1 = nu1_trajs_true[:,warmup-1,:]
nu2 = nu2_trajs_true[:,warmup-1,:]

meas.plot_measure(0.01 * nu1, (0,2), num_bins, 'hist')

#%% calculate distance between distributions for ESN trajectories
sigma_kernel = 1
kernel = lambda x,y: meas.rbf(x,y,sigma_kernel)
step = 10

#%%
dist_trajs_truepred_11 = np.zeros(T-warmup)
for t in range(warmup, T):
    if t % step == 0:
        nu_a_t, nu_b_t = nu1_trajs_true[:,t,:], nu1_trajs_pred[:,t,:]
        dist_trajs_truepred_11[t-warmup] = meas.MMD(nu_a_t, nu_b_t, kernel)
        print('time:', t)

plt.plot(dist_trajs_truepred_11)

#%%
dist_trajs_truepred_22 = np.zeros(T-warmup)
for t in range(warmup, T):
    if t % step == 0:
        nu_a_t, nu_b_t = nu2_trajs_true[:,t,:], nu2_trajs_pred[:,t,:]
        dist_trajs_truepred_22[t-warmup] = meas.MMD(nu_a_t, nu_b_t, kernel)
        print('time:', t)

plt.plot(dist_trajs_truepred_22)
#%%
dist_trajs_truepred_12 = np.zeros(T-warmup)
for t in range(warmup, T):
    if t % step == 0:
        nu_a_t, nu_b_t = nu1_trajs_true[:,t,:], nu2_trajs_pred[:,t,:]
        dist_trajs_truepred_12[t-warmup] = meas.MMD(nu_a_t, nu_b_t, kernel)
        print('time:', t)

plt.plot(dist_trajs_truepred_12)
#%%
dist_trajs_truepred_21 = np.zeros(T-warmup)
for t in range(warmup, T):
    if t % step == 0:
        nu_a_t, nu_b_t = nu2_trajs_true[:,t,:], nu1_trajs_pred[:,t,:]
        dist_trajs_truepred_21[t-warmup] = meas.MMD(nu_a_t, nu_b_t, kernel)
        print('time:', t)

plt.plot(dist_trajs_truepred_21)

#%%
nu1 = nu1_trajs_pred[:,-1,:]
meas.plot_measure(nu1, (0,2))

#%%
nu2 = nu2_trajs_pred[:,-1,:]
meas.plot_measure(nu2, (0,2))

#%%
x_plot = nu1[:,0]
y_plot = nu1[:,1]
z_plot = nu1[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#########################################################################################################
#%% compare the invariant measures of lorenz and ESN
# find invariant measure lorenz

n_init_cond_meas = 200
t_start = 300 # need to start from about 300 to make sure you are on the attractor
t_end = 600 # need a good length to cover the whole attractor, 600 seems sufficient
z0 = np.zeros((3))
sd_meas = 20

find_invar_meas = True
if find_invar_meas:
    lor = ds.lorenz()
    mu = meas.invariant_measure(lor,n_init_cond_meas, t_start, t_end, z0, sd_meas )
    np.save('Lorenz invariant measure', mu)
else:
    mu = np.load('Lorenz invariant measure.npy')
#%% plot invariant measure

meas.plot_measure(mu, (2,1), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better

#%% invariant measure ESN
N = network.reservoir_size
n_init_cond_meas = 300
t_start = 400 # need to start from about 300 to make sure you are on the attractor
t_end = 800 # need a good length to cover the whole attractor, 600 seems sufficient
x0 = np.zeros((N))
sd_meas = 300
model_ds = Model_DS(model)

find_invar_meas = True
if find_invar_meas:
    zeta_ESN = meas.invariant_measure(model_ds, n_init_cond_meas, t_start, t_end, x0, sd_meas )
    zeta_ESN = torch.from_numpy(zeta_ESN)
    mu_ESN = network.readout(zeta_ESN)
    mu_ESN = mu_ESN.detach().cpu().numpy()
    np.save('Lorenz ESN invariant measure state space', zeta_ESN)
    np.save('Lorenz ESN invariant measure readout', mu_ESN)
else:
    zeta_ESN = np.load('Lorenz ESN invariant measure state space.npy')
    mu_ESN = np.load('Lorenz ESN invariant measure readout.npy')

#%% plot invariant measure

meas.plot_measure(mu_ESN, (2,0), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better


#%%
x_plot = mu_ESN[:,0]
y_plot = mu_ESN[:,1]
z_plot = mu_ESN[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# %%
