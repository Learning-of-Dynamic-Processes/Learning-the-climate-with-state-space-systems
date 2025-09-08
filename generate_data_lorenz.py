#%%
import numpy as np
from utils.dynamical_systems import lorenz

step = 0.02

#%% generate training data
n_init_conds_train = 100
y0 = np.array([0,0,27]).astype(np.float64)
lor = lorenz()

init_conds_train = lor.generate_ics(n_init_conds_train, y0)

np.save('Lorenz_initial_conditions_train', init_conds_train)

#%%
T_train = 5000
trajectories_train = lor.integrate(init_conds_train, T_train)

np.save('Lorenz_trajectories_train', trajectories_train)

#%% generate testing data
n_init_conds_test = 100
y0 = np.array([0,0,27]).astype(np.float64)
lor = lorenz(step = step)

init_conds_test = lor.generate_ics(n_init_conds_test, y0)

np.save('Lorenz_initial_conditions_test', init_conds_test)

#%%
T_test = 5000
trajectories_test = lor.integrate(init_conds_test, T_test)

np.save('Lorenz_trajectories_test', trajectories_test)


# %%
