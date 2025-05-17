import torch.nn as nn
import torch.nn.functional as F
import importlib

from .categorical_stream import CategoricalStream

class FactorizedStream(nn.Module):
    def __init__(self, input_dim, value_stream_config, speed_stream_config, direction_stream_config, num_atoms):
        super().__init__()
        self.value_stream = CategoricalStream(input_dim, value_stream_config, num_atoms)
        self.speed_advantage_stream = CategoricalStream(input_dim, speed_stream_config, num_atoms)
        self.direction_advantage_stream = CategoricalStream(input_dim, direction_stream_config, num_atoms)

    def forward(self, x):
        batch_size = x.shape[0]

        value = self.value_stream(x).view(batch_size, 1, -1)  # [batch_size, 1, num_atoms]
        num_atoms = value.shape[2]

        # ストリームごとにQ値を計算
        speed_advantage = self.speed_advantage_stream(x).view(batch_size, -1, num_atoms)   # [batch_size, num_speeds, num_atoms]
        direction_advantage = self.direction_advantage_stream(x).view(batch_size, -1, num_atoms) # [batch_size, num_directions, num_atoms]

        speed_advantage = speed_advantage - speed_advantage.mean(dim=1, keepdim=True)
        direction_advantage = direction_advantage - direction_advantage.mean(dim=1, keepdim=True)

        # Q 値計算：独立したストリームごとに計算
        q_atoms_speed = value + speed_advantage  # [batch_size, num_speeds, num_atoms]
        q_atoms_direction = value + direction_advantage  # [batch_size, num_directions, num_atoms]

        # ソフトマックス適用
        probabilities_speed = F.softmax(q_atoms_speed, dim=2)
        probabilities_direction = F.softmax(q_atoms_direction, dim=2)

        return probabilities_speed, probabilities_direction