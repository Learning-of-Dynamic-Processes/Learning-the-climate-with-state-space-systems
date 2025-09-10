from multiprocessing import Pool

import numpy as np
import torch
import utils.dynamical_systems
from tqdm.auto import tqdm
import os

class Dataset:
    """Dataset of transients obtained from a given system."""

    def __init__(self, 
                 num_trajectories: int,
                 len_trajectories: int, 
                 step : float,
                 dynamical_system_name = 'lorenz', 
                 parameters = None, 
                 initial_point : np.ndarray = None, 
                 sigma : float = 1, 
                 data_type = None,
                 method = 'RK4',
                 load_data: bool = False,
                 data_set_name : str = ''
                 ) -> None:
        """
        Create set of trajectories.
        num_trajectories : num_trajectories
        len_trajectories : len_trajectories
        step : time_step
        parameters : for dynamical system
        initial_point : for generating initial conditions
        sigma : for generating initial conditions
        data_set_name : 'train', 'validate', 'test'
        """

        self.data_type = data_type
        self.dynamical_system_name = dynamical_system_name
        self.data_set_name = data_set_name
        self.path = dynamical_system_name + "/" + data_set_name
        if dynamical_system_name == 'lorenz':
            dynamical_system = utils.dynamical_systems.lorenz

        if load_data:
            self.tt = np.load(self.path + "/time_array.npy")
            self.input_data = np.load(self.path + "/input_data.npy")
            self.output_data = np.load(self.path + "/output_data.npy")
            self.ids = np.arange(len(self.input_data))
        else:
            print("Creating data")
            time_array = np.array([i*step for i in range(len_trajectories)])
            self.ids = np.arange(num_trajectories)
            ds = dynamical_system(step, parameters, method)
            init_conds = ds.generate_ics(num_trajectories, initial_point, sigma)
            trajectories = ds.integrate(init_conds, len_trajectories)

            self.input_data = trajectories[:, :-1, :]
            self.output_data = trajectories[:, 1:, :]
            self.tt = time_array

            folder = self.path
            os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

            np.save(os.path.join(folder, "time_array.npy"), self.tt)
            np.save(os.path.join(folder, "input_data.npy"), self.input_data)
            np.save(os.path.join(folder, "output_data.npy"), self.output_data)

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