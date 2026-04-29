from multiprocessing import Pool

import numpy as np
import torch
import utils.dynamical_systems as dyn_sys
from tqdm.auto import tqdm
import os
import utils.measures as meas
import matplotlib.pyplot as plt
import time

lor_args = (10, 8/3, 28)

def shift_scale(y, shift, scale):
    """
    y : (batch, seq_length, n_dim)
    shift : (n_dim)
    scale : (n_dim)
    """
    batch, seq_length, _ = y.shape
    shift = shift[np.newaxis, np.newaxis, :]
    scale = scale[np.newaxis, np.newaxis, :]

    return (y - shift) * scale

def undo_shift_scale(y, shift, scale):
    """
    y : (batch, seq_length, n_dim)
    shift : (n_dim)
    scale : (n_dim)
    """
    batch, seq_length, _ = y.shape
    shift = shift[np.newaxis, np.newaxis, :] 
    scale = scale[np.newaxis, np.newaxis, :]
    return y / scale + shift

def normalise(y):
    """
    y : (batch, seq_length, n_dim)
    """
    mean = y.mean(axis=(0,1), keepdims=True)
    centered = y # - mean

    max_abs = np.max(np.abs(centered), axis=(0,1), keepdims=True)
    scale = 1.0 / 100 # (5.0 * max_abs)

    mean = np.zeros(y.shape[2]) # mean[0,0,:]
    scale = np.array([1 /100 for _ in range(y.shape[2])]) # scale[0,0,:]

    return (shift_scale(y, mean, scale), mean, scale)

def downsample_array(arr, n_new_sample, axis=0, seed=None, replace=False):
    """
    Randomly downsample a numpy array along a given axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array of arbitrary dimension.
    n_new_sample : int
        Number of samples to select (must be <= arr.shape[axis] if replace=False).
    axis : int, default=0
        Axis along which to downsample.
    seed : int, optional
        Random seed for reproducibility. If None, a random generator is used.
    replace : bool, default=False
        Whether sampling is with replacement.

    Returns
    -------
    np.ndarray
        Downsampled array with shape modified along `axis`.
    """
    n_samples = arr.shape[axis]
    if not replace and n_new_sample > n_samples:
        raise ValueError("n_new_sample must be <= number of samples on the chosen axis")

    rng = np.random.default_rng(seed)  # independent RNG
    indices = rng.choice(n_samples, size=n_new_sample, replace=replace)

    return np.take(arr, indices, axis=axis)

class Dataset:
    """Dataset of transients obtained from a given system."""

    def __init__(self, 
                 num_trajectories: int,
                 len_trajectories: int, 
                 step : float = 1,
                 dynamical_system_name : str = 'lorenz', 
                 parameters = lor_args, 
                 initial_points_mean : np.ndarray = None, 
                 initial_points_sd : float = 1, 
                 data_type = torch.float64,
                 method : str = 'RK4',
                 load_data: bool = False,
                 data_set_name : str = '',
                 normalize_data : bool = True,
                 shift : np.ndarray = None,
                 scale : np.ndarray = None,
                 ) -> None:
        """
        Create set of trajectories.
        num_trajectories : num_trajectories
        len_trajectories : len_trajectories
        step : time_step for incrementing dynamical system
        parameters : for dynamical system
        initial_point : for generating initial conditions, shape (n_dim)
        sigma : for generating initial conditions
        data_set_name : 'train', 'validate', 'test'
        """

        self.data_type = data_type
        self.dynamical_system_name = dynamical_system_name
        self.data_set_name = data_set_name
        self.path = dynamical_system_name + "/" + data_set_name
        if dynamical_system_name == 'lorenz':
            dynamical_system = dyn_sys.lorenz
        else:
            raise ValueError(f"Dynamical system {dynamical_system_name} not supported.")


        if load_data:
            self.tt = np.load(self.path + "/time_array.npy")
            self.input_data = np.load(self.path + "/input_data.npy")
            self.output_data = np.load(self.path + "/output_data.npy")
            shift_scale_val = np.load(self.path + "/shift_scale.npy")
            self.shift, self.scale = shift_scale_val
            self.ids = np.arange(len(self.input_data))
        else:
            print("Creating data")
            time_array = np.arange(0, (len_trajectories+1)*step, step)
            self.ids = np.arange(num_trajectories)
            ds = dynamical_system(step, parameters, method)
            init_conds = dyn_sys.generate_points(num_trajectories, initial_points_mean, initial_points_sd)
            trajectories = ds.integrate(init_conds, len_trajectories + 1)

            self.input_data = trajectories[:, :-1, :] # (num_trajectories, len_trajectories, n_dim)
            self.output_data = trajectories[:, 1:, :] # (num_trajectories, len_trajectories, n_dim)
            self.tt = time_array # (len_trajectories)

            if normalize_data:
                if shift is None and scale is None:
                    _, shift, scale = normalise(self.input_data)
                self.shift = shift
                self.scale = scale
                self.input_data = shift_scale(self.input_data, shift, scale)
                self.output_data = shift_scale(self.output_data, shift, scale)

            folder = self.path
            os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

            np.save(os.path.join(folder, "time_array.npy"), self.tt)
            np.save(os.path.join(folder, "input_data.npy"), self.input_data)
            np.save(os.path.join(folder, "output_data.npy"), self.output_data)
            np.save(os.path.join(folder, "shift_scale.npy"), (self.shift, self.scale))


    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple:
        """Return a trajectory."""
        return torch.tensor(self.input_data[self.ids[index]], dtype=self.data_type), torch.tensor(
            self.output_data[self.ids[index]], dtype=self.data_type)

    def save_data(self) -> None:
        """Save the trajectories."""
        
        folder = self.path
        os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist
        np.savez(
            folder + '/data',
            input_data=self.input_data,
            output_data=self.output_data,
            tt_arr=self.tt,
            ids=self.ids,
        )

"""
class ParallelDataset # to be implemented
"""

class Two_Sample:
    """Time series of two samples over time"""
    def __init__(self,
                 mu1 : np.ndarray = None,
                 mu2 : np.ndarray = None,
                 load_data : bool = False,
                 load_dist_mmd : bool = False,
                 load_dist_wass1 : bool = False,
                 load_dist_wass1_traj : bool = False,
                 path : str = None,
                 name : str = None,
                 name1 : str = None,
                 name2 : str = None
                 ):
        """
        mu1 : (m_sample, T_seq_len, d_dim)
        mu2 : (n_sample, T_seqe_len, d_dim)
        """
        self.path = path
        self.name = name
        self.name1 = name1
        self.name2 = name2

        if load_data:
            self.load_data()
        else:
            self.mu1 = mu1
            self.mu2 = mu2

        if load_dist_mmd:
            self.load_dist_mmd()
        if load_dist_wass1:
            self.load_dists_wass1()
        if load_dist_wass1_traj:
            self.load_dists_wass1_traj()
        else:
            self.dist_mmd = None
            self.dists_wass1 = None
            self.dists_wass1_traj = None
        
    def load_data(self):
        print(self.path + self.name1 + '.npy')
        self.mu1 = np.load(self.path + self.name1 + '.npy')
        self.mu2 = np.load(self.path + self.name2 + '.npy')

    def load_dist_mmd(self):
        print(self.path + self.name + '_mmd.npy')
        self.dist_mmd = np.load(self.path + self.name + '_mmd.npy')
        
    def load_dists_wass1(self):
        self.dists_wass1 = np.load(self.path + self.name + '_wass1.npy')

    def load_dists_wass1_traj(self):
        self.dists_wass1_traj = np.load(self.path + self.name + '_wass1_traj.npy')

    def save_data(self):
        np.save(self.path + self.name1, self.mu1)
        np.save(self.path + self.name2, self.mu2)

    def save_dist_mmd(self):
        np.save(self.path + self.name + '_mmd', self.dist_mmd)

    def save_dists_wass1(self):
        np.save(self.path + self.name + '_wass1', self.dists_wass1)

    def save_dists_wass1_traj(self):
        np.save(self.path + self.name + '_wass1_traj', self.dists_wass1_traj)

    def plot_dists_mmd(self, plot_name):
        fig, ax = plt.figure(), plt.axes()
        ax.plot(self.dist_mmd)

        plt.savefig(self.path + self.name + plot_name + '_mmd')
        plt.close()
    
    def plot_dists_wass1(self, plot_name):
        fig, ax = plt.figure(), plt.axes()
        for i in range(self.dists_wass1.shape[0]):
            ax.plot(self.dists_wass1[i])

        plt.savefig(self.path + self.name + plot_name + '_wass1')
        plt.close()

    def plot_dists_wass1_traj(self, plot_name):
        fig, ax = plt.figure(), plt.axes()
        for i in range(self.dists_wass1_traj.shape[0]):
            ax.plot(self.dists_wass1_traj[i])

        plt.savefig(self.path + self.name + plot_name + '_wass1_traj')
        plt.close()

    def median_dist(self, n_estimate = None):
        mu = np.concatenate([self.mu1, self.mu2], axis = 0)
        
        if n_estimate is not None:
            idx = np.random.choice(mu.shape[0], n_estimate, replace = False)
            mu = mu[idx]

        xx = np.sum(mu**2, axis = -1)
        xy = np.einsum("mtd, ntd -> mnt", mu, mu)
        dist_sq = xx[:, None, :] + xx[None, :, :] - 2 * xy
        dists = np.sqrt(np.maximum(dist_sq, 0.0))

        mask = np.eye(mu.shape[0], dtype=bool)[:, :, None]
        dists = np.where(mask, np.nan, dists)

        medians = np.nanmedian(dists, axis=(0, 1))

        return np.mean(medians)

    def calculate_dist_mmd(self, 
                       sigma = 1.0,
                       biased = False,
                       linear_time = False,
                       enforce_equal = False):
        if enforce_equal:
            m, n = self.mu1.shape[0], self.mu2.shape[0]
            p = min(m, n)
            idx1 = np.random.choice(m, p, replace = False)
            idx2 = np.random.choice(n, p, replace = False)

            mu1 = self.mu1[idx1]
            mu2 = self.mu2[idx2]
        else:
            mu1 = self.mu1
            mu2 = self.mu2

        start_time = time.time()
        print("calculating distances")
        if linear_time:
            self.dist_mmd = meas.mmd_rbf_seq_lin_time(mu1, mu2, sigma)
        else:
            self.dist_mmd = meas.mmd_rbf_seq(mu1, mu2, sigma, biased)
        
        plot_name = "_biased_" + str(biased) + "_linear_time_" + str(linear_time) + '_mmd'
        self.plot_dists_mmd(plot_name)
        self.save_dist_mmd()
        elapsed_time = time.time() - start_time
        print(f"Time to calculate distances: {elapsed_time:.3f} seconds")
        
        return self.dist_mmd


    def calculate_dists_wass1(self,coord_list : list = None,
                       enforce_equal : bool = False):
        if enforce_equal:
            m, n = self.mu1.shape[0], self.mu2.shape[0]
            p = min(m, n)
            idx1 = np.random.choice(m, p, replace = False)
            idx2 = np.random.choice(n, p, replace = False)

            mu1 = self.mu1[idx1]
            mu2 = self.mu2[idx2]
        else:
            mu1 = self.mu1
            mu2 = self.mu2

        start_time = time.time()
        print("calculating distances")
        self.dists_wass1 = meas.wasserstein1_seq(mu1, mu2) # (d,T)
        
        plot_name = 'wass1'
        self.plot_dists_wass1(plot_name)
        self.save_dists_wass1()
        elapsed_time = time.time() - start_time
        print(f"Time to calculate distances: {elapsed_time:.3f} seconds")
        
        return self.dists_wass1
    
    def calculate_dists_wass1_traj(self,coord_list : list = None, init_cond : int = 0, warmup : int = 1000, aggregate_invar_meas : int = 1):
        n, T, d = self.mu1.shape
        invar_meas = self.mu1[:,T-aggregate_invar_meas:, : ].reshape(-1,d)
        trajectory = self.mu2[init_cond, warmup:, :]

        start_time = time.time()
        print("calculating distances")
        self.dists_wass1_traj = meas.wasserstein1_traj(invar_meas, trajectory)
        
        plot_name = 'wass1_traj'
        self.plot_dists_wass1_traj(plot_name)
        self.save_dists_wass1_traj()
        elapsed_time = time.time() - start_time
        print(f"Time to calculate distances: {elapsed_time:.3f} seconds")
        
        return self.dists_wass1_traj