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


class DQNPreCat(nn.Module):
    def __init__(self, dqn_config, device = "cpu"):
        super().__init__()

        self.input_adapters = []
        self.input_adapters.append(adapter.RnnInputAdapter(dqn_config["rnn"]))
        self.input_adapters.append(adapter.InputAdapter(dqn_config, device))
        
        factory = DQNFactory(dqn_config)
        self.streams = factory.create_streams()
        
        self.is_factorized = "speed_advantage_stream" in dqn_config and "direction_advantage_stream" in dqn_config

        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"]) if not self.is_factorized else adapter.QValueAdapterMultiDim(dqn_config["categorical"])
        self.action_adapter = adapter.ActionAdapter() if not self.is_factorized else adapter.ActionMultiDimAdapter()

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