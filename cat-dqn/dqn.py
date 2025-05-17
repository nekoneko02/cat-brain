import numpy as np
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

        self.has_rnn = "rnn" in dqn_config # RNN層がある
        if self.has_rnn: 
          hidden_dim = dqn_config["rnn"]["hidden_dim"]
          self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms) # [num_atom, ]

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
        self.is_factorized = (
          "speed_advantage_stream" in dqn_config and
          "direction_advantage_stream" in dqn_config
        )
        if self.is_factorized:
          self.speed_advantage_stream = self._generate_streams(medium_dim, dqn_config["speed_advantage_stream"])
          self.direction_advantage_stream = self._generate_streams(medium_dim, dqn_config["direction_advantage_stream"])
        else:
          self.advantage_stream = self._generate_streams(medium_dim, dqn_config["advantage_stream"])

        self.return_info = False

    def _generate_streams(self, input_dim, layer_configs):
        layers = []
        for layer in layer_configs[:-1]:
          layers.append(rlnn.NoisyLinear(input_dim, layer))
          layers.append(nn.ReLU())
          input_dim = layer
        layers.append(rlnn.NoisyLinear(input_dim, layer_configs[-1]*self.num_atoms))
        return nn.Sequential(*layers)
    
    def forward(self, x, hidden_state=None):
        if self.is_factorized:
          if self.return_info:
            # 分布取得
            probabilities_speed, probabilities_direction, hidden_state, speed_advantage, direction_advantage = self.forward_distribution(x, hidden_state)
            
            # 各ストリームごとの Q 値計算
            q_values_speed = torch.sum(probabilities_speed * self.get_support(), dim=-1)  # [batch_size, num_speeds]
            q_values_direction = torch.sum(probabilities_direction * self.get_support(), dim=-1)  # [batch_size, num_directions]
            
            return q_values_speed, q_values_direction, hidden_state, speed_advantage, direction_advantage

          else:
            # 分布取得
            probabilities_speed, probabilities_direction = self.forward_distribution(x, hidden_state)
            
            # 各ストリームごとの Q 値計算
            q_values_speed = torch.sum(probabilities_speed * self.get_support(), dim=-1)  # [batch_size, num_speeds]
            q_values_direction = torch.sum(probabilities_direction * self.get_support(), dim=-1)  # [batch_size, num_directions]
            
            return q_values_speed, q_values_direction
        else:
          if self.return_info:
            probabilities, hidden_state, speed_advantage, direction_advantage = self.forward_distribution(x, hidden_state)
            q_values = torch.sum(probabilities * self.get_support(), dim=-1)
            q_values_by_speed = torch.sum(speed_advantage * self.get_support(), dim=-1)
            q_values_by_direction = torch.sum(direction_advantage * self.get_support(), dim=-1)
            return q_values, hidden_state, q_values_by_speed, q_values_by_direction
          else:
            probabilities = self.forward_distribution(x, hidden_state)
            q_values = torch.sum(probabilities * self.get_support(), dim=-1)
            return q_values

    def forward_distribution(self, x, hidden_state=None):
        batch_size = x.shape[0]

        if self.has_rnn:
          # RNNの処理
          x, hidden_state = self.rnn(x, hidden_state)
          x = x[:, -1, :]  # 最後の出力のみを使用

        # 特徴抽出
        x = self.feature(x)
        value = self.value_stream(x).view(batch_size, 1, self.num_atoms)  # [batch_size, 1, num_atoms]
        if self.is_factorized:
          # ストリームごとにQ値を計算
          speed_advantage = self.speed_advantage_stream(x).view(batch_size, -1, self.num_atoms)   # [batch_size, num_speeds, num_atoms]
          direction_advantage = self.direction_advantage_stream(x).view(batch_size, -1, self.num_atoms) # [batch_size, num_directions, num_atoms]

          speed_advantage = speed_advantage - speed_advantage.mean(dim=1, keepdim=True)
          direction_advantage = direction_advantage - direction_advantage.mean(dim=1, keepdim=True)

          # Q 値計算：独立したストリームごとに計算
          q_atoms_speed = value + speed_advantage  # [batch_size, num_speeds, num_atoms]
          q_atoms_direction = value + direction_advantage  # [batch_size, num_directions, num_atoms]

          # ソフトマックス適用
          probabilities_speed = F.softmax(q_atoms_speed, dim=2)
          probabilities_direction = F.softmax(q_atoms_direction, dim=2)

          if self.return_info:
              return probabilities_speed, probabilities_direction, hidden_state, speed_advantage, direction_advantage
          else:
              return probabilities_speed, probabilities_direction
        else:
          advantage = self.advantage_stream(x).view(batch_size, -1, self.num_atoms)
          advantage = advantage - advantage.mean(dim=1, keepdim=True)

          q_atoms = value + advantage

          # Apply softmax to get probabilities
          probabilities = F.softmax(q_atoms, dim=2)

          if self.return_info:
            return probabilities, hidden_state, speed_advantage, direction_advantage
          else:
            return probabilities # [batch_size, output_dim, num_atoms]

    def get_support(self):
        return self.z_support
