import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

import stream
importlib.reload(stream)
import adapter
importlib.reload(adapter)
import dqn_factory
importlib.reload(dqn_factory)




class DQNOpticalCat(nn.Module):
    def __init__(self, dqn_config, device = "cpu"):
        super().__init__()

        self.input_adapters = []
        self.input_adapters.append(adapter.RnnInputAdapter(dqn_config["rnn"]))
        self.input_adapters.append(adapter.InputAdapter(dqn_config, device))

        factory = dqn_factory.DQNFactory(dqn_config)
        self.streams = factory.create_streams()
        self.is_factorized = False

        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"]) if not self.is_factorized else adapter.QValueAdapterMultiDim(dqn_config["categorical"])
        self.action_adapter = adapter.ActionAdapter() if not self.is_factorized else adapter.ActionMultiDimAdapter()
        self.temperature = dqn_config["temperature"]

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
        self.pre_cat.reset_noise()
