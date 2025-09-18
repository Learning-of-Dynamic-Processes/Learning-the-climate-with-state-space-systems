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
    
RC_type = config["MODEL"]["RC_type"]

tag = RC_type + '_ridge_' + str(config["TRAINING"]["ridge"])

if RC_type == 'ESN':
    Network = ESN
    Model = ESNModel
elif RC_type == 'RCN':
    Network = RCN
    Model = RCNModel
else:
    print('RC not supported')

#%%
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
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'train',
    normalize_data = config["DATA"]["normalize_data"]
)
shift, scale = dataset_train.shift, dataset_train.scale
dataset_train.save_data()

#%%
dataset_val = Dataset(
    num_trajectories = config["DATA"]["n_val"],
    len_trajectories = config["DATA"]["l_trajectories"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'validate',
    normalize_data = config["DATA"]["normalize_data"],
    shift = shift,
    scale = scale
)
dataset_val.save_data()

#%%
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


#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 0], label="u")
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 1], label="v")
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 2], label="w")
ax.set_xlabel("t")
plt.legend()

folder = dynamical_system_name + "/fig"
os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

plt.savefig(os.path.join(folder, "data.png"))
plt.close()

#%%
# Create PyTorch dataloaders for train and validation data
dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

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
    dataloader_train,
    dataloader_val,
    network,
    learning_rate=config["TRAINING"]["learning_rate"],
    offset=config["TRAINING"]["offset"],
    ridge_factor=config["TRAINING"]["ridge_factor"],
    device=config["TRAINING"]["device"],
)
#%%
config["TRAINING"]["ridge"]

#%%
if config["TRAINING"]["ridge"]:
    model.train(ridge=config["TRAINING"]["ridge"])
else:
    # Train for the given number of epochs
    progress_bar = tqdm(
        range(0, config["TRAINING"]["epochs"]),
        leave=True,
        position=0,
        desc=progress(0, 0),
    )
    train_loss_list = []
    val_loss_list = []
    for _ in progress_bar:
        train_loss = model.train(ridge=config["TRAINING"]["ridge"])
        val_loss = model.validate()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        progress_bar.set_description(progress(train_loss, val_loss))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_list, label="train loss")
    ax.plot(val_loss_list, label="val loss")
    plt.legend()
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("")
    plt.show()

model.net = model.net.to("cpu")
model.save_network(config["PATH"] + tag + "_model_")
model.net = model.net.to(model.device)

#%%
warmup = config["DATA"]["max_warmup"]
predictions, _ = model.integrate(
    torch.tensor(dataset_test.input_data[0, :warmup, :], dtype=torch.get_default_dtype()).unsqueeze(0).to(model.device),
    T=dataset_test.input_data[0].shape[0] - warmup,
)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[0][:, 0], label="true")
if len(predictions.shape) > 1:
    ax.plot(dataset_test.tt[:-1], predictions[:, :, 0].detach().squeeze(0), label="prediction")
else:
    ax.plot(dataset_test.tt[:-1], predictions, label="prediction")
ax.axvline(x=dataset_test.tt[warmup], color="k")
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
folder = dynamical_system_name + "/fig"
os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

plt.savefig(os.path.join(folder, "predictions.png"))
plt.close()
# %%