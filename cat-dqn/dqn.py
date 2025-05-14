import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.modules as rlnn

class DQN(nn.Module):
    def __init__(self, dqn_config):
        super().__init__()

        input_dim = dqn_config["input_dim"]
        categorical_config = dqn_config["categorical"]
        self.num_atoms = categorical_config["num_atoms"]
        self.v_min, self.v_max = categorical_config["v_min"], categorical_config["v_max"]
        if "rnn" in dqn_config: # RNN層
          self.has_rnn = True
          hidden_dim = dqn_config["rnn"]["hidden_dim"]
          self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
          self.has_rnn = False

        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        # 特徴抽出層
        layers = []
        for layer in dqn_config["feature"]:
          layers.append(nn.LazyLinear(layer))
          layers.append(nn.ReLU())
        self.feature = nn.Sequential(*layers)

        medium_dim = dqn_config["feature"][-1]
        # 価値関数
        self.value_stream = self._generate_streams(medium_dim, dqn_config["value_stream"])
        # アドバンテージ関数 A(s, a)
        self.advantage_stream = self._generate_streams(medium_dim, dqn_config["advantage_stream"])

    def _generate_streams(self, input_dim, layer_configs):
        layers = []
        for layer in layer_configs[:-1]:
          layers.append(rlnn.NoisyLinear(input_dim, layer))
          layers.append(nn.ReLU())
          input_dim = layer
        layers.append(rlnn.NoisyLinear(input_dim, layer_configs[-1]*self.num_atoms))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.has_rnn:
          probability, _ = self._forward_with_hidden_state(x, None)
        else:
          probability = self._forward_with_hidden_state(x, None)
        return probability

    def _forward_with_hidden_state(self, x, hidden_state=None):
        batch_size = x.shape[0]

        if self.has_rnn:
          # RNNの処理
          x, hidden_state = self.rnn(x, hidden_state)
          x = x[:, -1, :]  # 最後の出力のみを使用

        # 特徴抽出
        x = self.feature(x)
        value = self.value_stream(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(batch_size, -1, self.num_atoms)

        # Distributional Q-values
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_atoms = q_atoms.view(batch_size, -1, self.num_atoms)

        # Apply softmax to get probabilities
        probabilities = F.softmax(q_atoms, dim=2)

        if self.has_rnn:
          return probabilities, hidden_state
        else:
          return probabilities # [batch_size, output_dim, num_atoms]

    def get_support(self):
        return self.z_support
