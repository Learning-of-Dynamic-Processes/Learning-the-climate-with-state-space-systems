import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from tqdm.auto import tqdm
from utils.dynamical_systems import DS


class DenseStack(nn.Module):
    """
    Fully connected neural network.

    Args:
        config: Configparser section proxy with:
            num_in_features: Number of input features
            num_out_features: Number of output features
            num_hidden_features: List of nodes in each hidden layer
            use_batch_norm: If to use batch norm
            dropout_rate: If, and with which rate, to use dropout
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.fc_layers = []
        self.acts = []

        in_features = input_size
        # List containing number of hidden and output neurons
        list_of_out_features = [*hidden_size, output_size]
        for out_features in list_of_out_features:
            # Add fully connected layer
            self.fc_layers.append(nn.Linear(in_features, out_features))
            # Add activation function
            self.acts.append(nn.GELU())
            in_features = out_features
            self.num_out_features = out_features

        # Transform to pytorch list modules
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.acts = nn.ModuleList(self.acts)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fully connected neural network.

        Args:
            input_tensor: Tensor with input features

        Returns:
            Output prediction tensor
        """
        for i_layer in range(len(self.fc_layers)):
            # Fully connected layer
            input_tensor = self.fc_layers[i_layer](input_tensor)
            # Apply activation function, but not after last layer
            if i_layer < len(self.fc_layers) - 1:
                input_tensor = self.acts[i_layer](input_tensor)
        return input_tensor


class ESN(nn.Module):
    """Taken from https://github.com/danieleds/TorchRC/blob/master/torch_rc/nn/esn.py."""

    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        hidden_size: list,
        output_size: int,
        scale_rec: float = 1 / 1.1,
        scale_in: float = 1.0 / 40.0,
        leaking_rate: float = 0.5,
        rec_rescaling_method: str = "specrad",  # Either "norm" or "specrad"
    ):
        super(ESN, self).__init__()

        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size

        self.leaking_rate = leaking_rate

        # Reservoir
        W_in = torch.rand((reservoir_size, input_size)) - 1 / 2.0
        W_hat = torch.rand((reservoir_size, reservoir_size)) - 1 / 2.0

        W_in = scale_in * W_in
        W_hat = self.rescale_contractivity(W_hat, scale_rec, rec_rescaling_method)

        # Assign as buffers
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_hat", W_hat)
        # self.readout = nn.Linear(reservoir_size, output_size, bias=True)
        self.readout = DenseStack(reservoir_size, hidden_size, output_size)

    @staticmethod
    def rescale_contractivity(W, coeff, rescaling_method):
        if rescaling_method == "norm":
            return W * coeff / W.norm()
        elif rescaling_method == "specrad":
            return W * coeff / (torch.linalg.eig(W)[0].abs().max())
        else:
            raise Exception("Invalid rescaling method used (must be either 'norm' or 'specrad')")

    def forward_reservoir(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)
        output: (batch, hidden_size)
        """
        h_tilde = torch.tanh(torch.mm(input, self.W_in.t()) + torch.mm(hidden, self.W_hat.t()))
        h = (1 - self.leaking_rate) * hidden + self.leaking_rate * h_tilde
        return h

    def forward(self, input, h_0=None, return_states=False):
        """
        input : (batch, sequence_length, input_size)
        h_0 : (batch, hidden_size)
        """
        batch = input.shape[0]

        if h_0 is None:
            h_0 = input.new_zeros((batch, self.reservoir_size))

        next_layer_input = input  # (batch, sequence_length, input_size)
        layer_outputs = []  # list of (batch, hidden_size)
        step_h = h_0
        for i in range(next_layer_input.shape[1]):
            x_t = next_layer_input[:, i]
            h = self.forward_reservoir(x_t, step_h)  # (batch, hidden_size)
            step_h = h
            if return_states:
                layer_outputs.append(h)
            else:
                layer_outputs.append(self.readout(h))
        h_n = step_h
        layer_outputs = torch.stack(layer_outputs, axis=1)
        return layer_outputs, h_n


class ESNModel:
    def __init__(
        self, dataloader_train, dataloader_val, network, learning_rate=0.05, offset=1, ridge_factor=5e-9, device=None
    ):
        if torch.cuda.is_available() and device is None:
            self.device = "cuda"
        elif not torch.cuda.is_available() and device is None:
            self.device = "cpu"
        else:
            self.device = device

        print("Using:", self.device)

        self.net = network.to(self.device)

        self.trainable_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print("Trainable parameters: " + str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.offset = offset
        self.ridge_factor = ridge_factor

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.criterion = torch.nn.MSELoss().to(self.device)

        self.train_loss = []
        self.val_loss = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, min_lr=0.000001
        )

    def train(self, ridge=False):
        """Train model."""
        if ridge:
            x, y = torch.tensor(self.dataloader_train.dataset.input_data, dtype=torch.float64), torch.tensor(
                self.dataloader_train.dataset.output_data, dtype=torch.float64
            )
            out, _ = self.net(x.to(self.device), return_states=True)
            out = out[:, self.offset :]
            y = y[:, self.offset :]
            out_np = out.reshape(-1, out.shape[-1]).detach().cpu().numpy()
            y_np = y.reshape(-1, self.net.input_size).detach().cpu().numpy()

            clf = Ridge(alpha=self.ridge_factor)
            clf.fit(out_np, y_np)
            self.net.readout.fc_layers[0].weight = torch.nn.Parameter(
                torch.tensor(clf.coef_, dtype=torch.float64).to(self.device)
            )
            self.net.readout.fc_layers[0].bias = torch.nn.Parameter(
                torch.zeros_like(self.net.readout.fc_layers[0].bias).to(self.device)
            )
            sum_loss = self.criterion(self.net.readout(out), y.to(self.device)).detach().cpu().numpy()
            cnt = 1
        else:
            self.net.train()
            cnt, sum_loss = 0, 0
            for (x, y) in self.dataloader_train:
                self.optimizer.zero_grad()
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out[:, self.offset :], y[:, self.offset :].to(self.device))
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
            self.optimizer.zero_grad()
            self.scheduler.step(sum_loss / cnt)
        self.train_loss.append(sum_loss / cnt)
        return sum_loss / cnt

    def validate(self):
        """Validate model."""
        self.net.eval()
        cnt, sum_loss = 0, 0
        with torch.no_grad():
            for (x, y) in self.dataloader_val:
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out[:, self.offset :], y[:, self.offset :].to(self.device))
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
        self.val_loss.append(sum_loss / cnt)
        return sum_loss / cnt

    def integrate(self, x_0, T, h0=None):
        """
        Integrate batch of trajectories.
        x_0 : (batch, warm_up_length, input_size)
        T : num_time_steps_integrate_forward
        h0 : (batch, reservoir_size)
        """
        self.net.eval()

        batch = x_0.shape[0]
        warm_up_length = x_0.shape[1]
        print(batch, warm_up_length)
        if h0 is None:
            h0 = torch.zeros(batch, self.net.reservoir_size).to(self.device)

        x_trajectory = torch.zeros(batch, warm_up_length + T, self.net.input_size) # input_size = output_size for autoregressive integration
        h_trajectory = torch.zeros(batch, warm_up_length + T, self.net.reservoir_size)
        x_0 = x_0.to(self.device)
        
        # Warmup
        x_trajectory[:, :warm_up_length, :] = x_0
        h_trajectory[:, :warm_up_length, :], _ = self.net(x_0, h0, return_states = True)
        
        # Autoregressive integration
        x_t = x_trajectory[:, warm_up_length - 1, :].unsqueeze(1)
        h_t = h_trajectory[:, warm_up_length - 1, :]
        for t in tqdm(range(T), position=0, leave=True):
            x_out, h_out = self.net.forward(x_t, h_0 = h_t)
            x_t, h_t = x_out, h_out
            x_trajectory[:, t + warm_up_length, :] = x_t.squeeze(1)
            h_trajectory[:, t + warm_up_length, :] = h_t

        return x_trajectory, h_trajectory
    
    def Phi(self, h, t = None):
        """
        h : (batch, reservoir_size)
        """
        self.net.eval()
        input_is_numpy = isinstance(h, np.ndarray)
        
        if input_is_numpy:
            # Convert to tensor with same dtype
            orig_dtype = h.dtype
            h = torch.from_numpy(h).to(next(self.net.parameters()).device)

        x = self.net.readout(h) # (batch, input_size) (only works when inputs and outputs are same size)
        _, h_next = self.net(x.unsqueeze(1), h)

        if input_is_numpy:
                # Convert back to original dtype and NumPy
                h_next = h_next.detach().cpu().numpy().astype(orig_dtype)

        return h_next
    
    def save_network(self, name):
        """Save network weights and training loss history."""
        filename = name + "_reservoir_size_" + str(self.net.reservoir_size) + ".net"
        torch.save(self.net.state_dict(), filename)
        np.save(name + "_training_loss.npy", np.array(self.train_loss))
        np.save(name + "_validation_loss.npy", np.array(self.val_loss))
        return name

    def load_network(self, name):
        """Load network weights and training loss history."""
        filename = name + "_reservoir_size_" + str(self.net.reservoir_size) + ".net"
        self.net.load_state_dict(torch.load(filename))
        self.train_loss = np.load(name + "_training_loss.npy").tolist()
        self.val_loss = np.load(name + "_validation_loss.npy").tolist()


class ESNModel_DS(DS):
    def __init__(self, model):
        super().__init__(model.Phi, model.net.reservoir_size)

class RCN(ESN):
    
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        hidden_size: list,
        output_size: int,
        scale_rec: float = 1 / 1.1,
        scale_in: float = 1.0 / 40.0,
        leaking_rate: float = 0.5,
        rec_rescaling_method: str = "specrad",  # Either "norm" or "specrad"
    ):
        super(RCN, self).__init__(input_size, reservoir_size, hidden_size, output_size, scale_rec, scale_in, leaking_rate, rec_rescaling_method)

        self.readout = DenseStack(2*reservoir_size, hidden_size, output_size)

    def forward(self, input, h_0=None, return_states=False):
        """
        input : (batch, sequence_length, input_size)
        h_0 : (batch, hidden_size)
        """
        batch = input.shape[0]

        if h_0 is None:
            h_0 = input.new_zeros((batch, self.reservoir_size))

        next_layer_input = input  # (batch, sequence_length, input_size)
        layer_outputs = []  # list of (batch, hidden_size)
        step_h = h_0
        h_aug = h_0.new_zeros(batch, 2*self.reservoir_size)
        for i in range(next_layer_input.shape[1]):
            x_t = next_layer_input[:, i]
            h = self.forward_reservoir(x_t, step_h)  # (batch, hidden_size)
            if return_states:
                layer_outputs.append(h)
            else:
                # readout is the map (h_t-1, h_t) -> y_t
                h_aug[:, :self.reservoir_size] = step_h
                h_aug[:, self.reservoir_size: ] = h
                layer_outputs.append(self.readout(h_aug))
            step_h = h
        h_n = step_h
        layer_outputs = torch.stack(layer_outputs, axis=1)
        return layer_outputs, h_n


class RCNModel(ESNModel):
    def __init__(
        self,
        dataloader_train,
        dataloader_val,
        network,
        learning_rate=0.05,
        offset=1,
        ridge_factor=5e-9,
        device=None
    ):
        super(RCNModel, self).__init__(dataloader_train, dataloader_val, network, learning_rate, offset, ridge_factor, device)

    def train(self, ridge=False):
        """Train model."""
        if ridge:
            x, y = torch.tensor(self.dataloader_train.dataset.input_data, dtype=torch.float64), torch.tensor(
                self.dataloader_train.dataset.output_data, dtype=torch.float64
            )
            _out, _ = self.net(x.to(self.device), return_states=True) # (batch, sequence_length + offset, hidden_size)
            _out = _out[:, self.offset :]
            y = y[ :, self.offset + 1 :]

            # readout is trained  to learn (h_t-1, h_t) -> y_t
            batch, sequence_length = _out.shape[0], _out.shape[1]
            out = _out.new_zeros(batch, sequence_length-1, 2*self.net.reservoir_size)
            out[:, :, :self.net.reservoir_size] = _out[:, :-1, :]
            out[:, :, self.net.reservoir_size:] = _out[:, 1:, :]
            out_np = out.reshape(-1, out.shape[-1]).detach().cpu().numpy()
            y_np = y.reshape(-1, self.net.input_size).detach().cpu().numpy()

            print(out_np.shape)
            print(y_np.shape)

            clf = Ridge(alpha=self.ridge_factor)
            clf.fit(out_np, y_np)
            self.net.readout.fc_layers[0].weight = torch.nn.Parameter(
                torch.tensor(clf.coef_, dtype=torch.float64).to(self.device)
            )
            self.net.readout.fc_layers[0].bias = torch.nn.Parameter(
                torch.zeros_like(self.net.readout.fc_layers[0].bias).to(self.device)
            )
            sum_loss = self.criterion(self.net.readout(out), y.to(self.device)).detach().cpu().numpy()
            cnt = 1
        else:
            self.net.train()
            cnt, sum_loss = 0, 0
            for (x, y) in self.dataloader_train:
                self.optimizer.zero_grad()
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out[:, self.offset :], y[:, self.offset :].to(self.device))
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
            self.optimizer.zero_grad()
            self.scheduler.step(sum_loss / cnt)
        self.train_loss.append(sum_loss / cnt)
        return sum_loss / cnt

    def Phi(self, h, t = None):
        """
        h = [h_0, h_1] : (batch, 2*reservoir_size)
        """
        self.net.eval()
        input_is_numpy = isinstance(h, np.ndarray)

        if input_is_numpy:
            # Convert to tensor with same dtype
            orig_dtype = h.dtype
            h = torch.from_numpy(h).to(next(self.net.parameters()).device)
            batch = h.shape[0]

        h_1 = h[:, self.net.reservoir_size:]
        x = self.net.readout(h) # (batch, input_size) (only works when inputs and outputs are same size)

        h_aug = h.new_zeros(batch, 2*self.net.reservoir_size)
        h_aug[:, :self.net.reservoir_size] = h_1
        _, h_aug[:, self.net.reservoir_size:]= self.net(x.unsqueeze(1), h_1)

        if input_is_numpy:
            # Convert back to original dtype and NumPy
            h_aug = h_aug.detach().cpu().numpy().astype(orig_dtype)

        return h_aug
    
class RCNModel_DS(DS):
    def __init__(self, model):
        super().__init__(model.Phi, 2*model.net.reservoir_size)



def progress(train_loss, val_loss):
    """Define progress bar description."""
    return "Train/Loss: {:.6f}  Val/Loss: {:.6f}".format(train_loss, val_loss)