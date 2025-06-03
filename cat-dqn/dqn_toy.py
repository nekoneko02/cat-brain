import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

import dqn_factory
importlib.reload(dqn_factory)
import stream
importlib.reload(stream)
from dqn_factory import DQNFactory
import adapter
importlib.reload(adapter)


class DQNToy(nn.Module):
    def __init__(self, dqn_config, device = "cpu"):
        super().__init__()

        self.input_adapters = []
        self.input_adapters.append(adapter.InputAdapter(dqn_config, device))

        factory = DQNFactory(dqn_config)
        self.streams = factory.create_streams()
        
        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"])
        self.action_adapter = adapter.ActionAdapter()
        self.is_factorized = False

    def forward(self, x):
        for stream in self.streams:
            x = stream(x)
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
    
    def reset_noise(self):
        for st in self.streams:
            if isinstance(st, stream.DuelingStream):
                st.reset_noise()