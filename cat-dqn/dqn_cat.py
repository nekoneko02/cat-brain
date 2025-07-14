import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.modules as rlnn
from torchrl.modules.tensordict_module.actors import DistributionalQValueActor
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

import adapter
importlib.reload(adapter)

class DQNCat(nn.Module):
    def __init__(self, dqn_config, device="cpu"):
        super().__init__()
        self.device = device
        self.dqn_config = dqn_config

        # Adapter定義
        self.input_adapters = nn.ModuleList([
            adapter.RnnInputAdapter(dqn_config["rnn"]),
            adapter.InputAdapter(dqn_config, device)
        ])

        # ストリーム構築
        streams = []
        input_dim = dqn_config["input_dim"]

        hidden_dim = dqn_config["rnn"]["hidden_dim"]
        self.rnn = (nn.GRU(input_dim, hidden_dim, batch_first=True))

        for layer in dqn_config["feature"]:
            streams.append(nn.LazyLinear(layer))
            streams.append(nn.ReLU())

        medium_dim = dqn_config["feature"][-1]

        value_stream_config = dqn_config["value_stream"]
        advantage_stream_config = dqn_config["advantage_stream"]
        num_atoms = dqn_config["categorical"]["num_atoms"]
        self.value_stream = CategoricalStream(medium_dim, value_stream_config, num_atoms)
        self.advantage_stream = CategoricalStream(medium_dim, advantage_stream_config, num_atoms)

        self.streams = nn.ModuleList(streams)
        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"])  # 分布→期待値用
        self.action_adapter = adapter.ActionAdapter()
        self.temperature = dqn_config.get("temperature", 1.0)

    def forward(self, obs):  # obs: torch.Tensor [batch_size, seq, obs_dim]
        x = obs
        x = self._forward_rnn(x)
        for stream in self.streams:
            x = stream(x)
        x = self._forward_dueling(x)
        return x

    def to_input(self, x):
        for adapter in self.input_adapters:
            x = adapter(x)
        return x

    def to_action(self, probabilities):
        q_values = self.q_value_adapter(probabilities)
        actions = self.action_adapter(q_values)
        return actions

    def get_support(self):
        return self.q_value_adapter.z_support
    
    def _forward_rnn(self, x):
        x, _ = self.rnn(x, None)
        return x[:,-1,:]
        
    def _forward_dueling(self, x):
        value_output = self.value_stream(x)
        advantage_output = self.advantage_stream(x)
        q_atoms = value_output + advantage_output - advantage_output.mean(dim=1, keepdim=True)
        probabilities = F.softmax(q_atoms, dim=2)
        return probabilities
    

class CategoricalStream(nn.Module):
    def __init__(self, input_dim, layer_configs, num_atoms):
        super().__init__()
        self.num_atoms = num_atoms
        layers = []
        for layer in layer_configs[:-1]:
            layers.append(rlnn.NoisyLinear(input_dim, layer))
            layers.append(nn.ReLU())
            input_dim = layer
        layers.append(rlnn.NoisyLinear(input_dim, layer_configs[-1] * num_atoms))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.network(x).view(batch_size, -1, self.num_atoms)
        return output
    
    def reset_noise(self):
        for layer in self.network:
            if isinstance(layer, rlnn.NoisyLinear):
                layer.reset_noise()