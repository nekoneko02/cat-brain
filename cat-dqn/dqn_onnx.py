import torch.nn as nn

import importlib

import dqn
importlib.reload(dqn)

class DQNOnnx(nn.Module):
    def __init__(self, dqn):
        super().__init__()

        self.dqn = dqn

    def forward(self, x):
        x = self.dqn(x)
        q_values = self.dqn.q_value_adapter(x)
        q_values_speed, q_values_direction = q_values
        action = self.dqn.action_adapter(q_values)
        return action, q_values