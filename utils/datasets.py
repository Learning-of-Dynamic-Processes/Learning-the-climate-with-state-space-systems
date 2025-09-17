from multiprocessing import Pool

import numpy as np
import torch
import utils.dynamical_systems as dyn_sys
from tqdm.auto import tqdm
import os

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
    centered = y - mean

    max_abs = np.max(np.abs(centered), axis=(0,1), keepdims=True)
    scale = 1.0 / (5.0 * max_abs)

    mean = mean[0,0,:]
    scale = scale[0,0,:]

    return (shift_scale(y, mean, scale), mean, scale)

class Dataset:
    """Dataset of transients obtained from a given system."""

    def __init__(self, 
                 num_trajectories: int,
                 len_trajectories: int, 
                 step : float = 1,
                 dynamical_system_name : str = 'lorenz', 
                 parameters = lor_args, 
                 initial_point : np.ndarray = None, 
                 sigma : float = 1, 
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
            init_conds = dyn_sys.generate_points(num_trajectories, initial_point, sigma)
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