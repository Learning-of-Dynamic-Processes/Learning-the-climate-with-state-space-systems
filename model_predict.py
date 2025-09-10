#%%
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.datasets import Dataset
from utils.model import ESN, ESNModel, RCN, RCNModel, progress

dynamical_system_name = 'lorenz'

if dynamical_system_name == 'lorenz':
    from lorenz.config import config

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

RC_type = 'RCN'

if RC_type == 'ESN':
    Network = ESN
    Model = ESNModel
elif RC_type == 'RCN':
    Network = RCN
    Model = RCNModel
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

model.load_network(config["PATH"] + "model_")

#%% predict
warmup = config["DATA"]["max_warmup"]
predictions, _ = model.integrate(
    torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
    T=dataset_test.input_data.shape[1] - warmup,
)

# %%
