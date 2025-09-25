
#%%
nu1 = nu1_trajs_pred[:,warmup,:]
meas.plot_measure(nu1, (0,2), num_bins, "hist")

#%%
nu2 = nu2_trajs_true[:,warmup,:]
meas.plot_measure(nu2, (0,2), num_bins, "hist")

#%%
nu1 = nu1_trajs_pred[:,-1,:]
meas.plot_measure(nu1, (0,2), num_bins, "hist")

#%%
nu2 = nu2_trajs_true[:,-1,:]
meas.plot_measure(nu2, (0,2), num_bins, "hist")

#%%
nu1, nu2 = np.expand_dims(nu1_trajs_true[:, warmup, :], axis=1), np.expand_dims(nu2_trajs_pred[:, warmup, :], axis=1)
d1 = meas.mmd_rbf_seq(nu1, nu2)
print(d1)

#%%
nu1, nu2 = np.expand_dims(nu1_trajs_true[:, -1, :], axis=1), np.expand_dims(nu2_trajs_pred[:, -1, :], axis=1)
d2 = meas.mmd_rbf_seq(nu1, nu2)
print(d2)

#%%
d1b, d2b = dist_trajs_truepred_12[warmup], dist_trajs_truepred_12[-1]
print(d1b, d2b)
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

meas.plot_measure(mu_ESN, (1,2), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better


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

#%%
ids_off_attractor = []
ids_on_attractor = []
for i in range(mu_ESN.shape[0]):
    if mu_ESN[i,2]<0:
        ids_off_attractor.append(i)
    else:
        ids_on_attractor.append(i)
print(len(ids_off_attractor))
print(mu_ESN.shape[0])

# %%
mu_ESN_2 = mu_ESN[ids_on_attractor, :]
meas.plot_measure(mu_ESN_2, (1,2), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better


#%%
x_plot = mu_ESN_2[:,0]
y_plot = mu_ESN_2[:,1]
z_plot = mu_ESN_2[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# %%

#%%
import numpy as np

#%%
# Biased H0 ==
reject = .4
alpha = 0.05
sample_size = (2/reject) * (1 + np.sqrt(-2 * np.log(alpha)))**2
sample_size
# 60
# %%
# Unbiased H0 ==
reject = 0.4
alpha = 0.05
sample_size = - (4/reject)**2 * np.log(alpha)
sample_size

# 300
# %%
# Biased H0
bd = .25
alpha = 0.05
sample_size = (2/bd)**2 * (2 + np.sqrt( - np.log(alpha/2)))**2
sample_size
# %%

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
    samples_12.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    samples_12.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = False, enforce_equal=True)
    samples_12.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = True, enforce_equal=True)


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
    samples_11.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    samples_11.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = False, enforce_equal=True)
    samples_11.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = True, enforce_equal=True)
# dist_trajs_truepred_11 = samples_11.dist
#%% calculate distance between distributions for ESN trajectories
sigma_kernel = 1

dist_trajs_truepred_11 = meas.mmd_rbf_seq(nu1_trajs_true, nu1_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_11)

np.save(os.path.join(folder, "dist_trajs_truepred_11"+ tag), dist_trajs_truepred_11)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_11"+ tag))
plt.close()

#%%
dist_trajs_truepred_22 = meas.mmd_rbf_seq(nu2_trajs_true, nu2_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_22)

np.save(os.path.join(folder, "dist_trajs_truepred_22"+ tag), dist_trajs_truepred_22)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_22"+ tag))
plt.close()

#%%
dist_trajs_truepred_12 = meas.mmd_rbf_seq(nu1_trajs_true, nu2_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_12)

np.save(os.path.join(folder, "dist_trajs_truepred_12"+ tag), dist_trajs_truepred_12)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_12"+ tag))
plt.close()

#%%
dist_trajs_truepred_21 = meas.mmd_rbf_seq(nu2_trajs_true, nu1_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_21)

np.save(os.path.join(folder, "dist_trajs_truepred_21"+ tag), dist_trajs_truepred_21)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_21"+ tag))
plt.close()


#%%
sample_size_cut_off_two_sample_test(0.05, 0.05, '>eps', 0.01, False, 1)

# %%
two_sample_test(1000, 0.05, '>eps', 0.02, True, 1)
# %%
meas.two_sample_test(m, alpha = 0.05, H0 = '>eps', epsilon_sq = 0.15, biased = True)

#%%
meas.epsilon_sq_given_cut_off_two_sample_test(1000, 0.01, 0.05, '>eps', True)
# %%