import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

import dqn_factory
importlib.reload(dqn_factory)
from dqn_factory import DQNFactory
import adapter
importlib.reload(adapter)


class DQN(nn.Module):
    def __init__(self, dqn_config, device):
        super().__init__()

        self.input_adapters = []
        if "rnn" in dqn_config:
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

    def get_action(self, observation):
        x = observation
        for adapter in self.input_adapters:
            x = adapter(x)
        with torch.no_grad():
            probabilities = self.forward(x)
            q_values = self.q_value_adapter(probabilities)
            actions = self.action_adapter(q_values)
        return actions
    def get_support(self):
        return self.q_value_adapter.z_support