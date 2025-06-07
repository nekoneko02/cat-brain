import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib

import stream
importlib.reload(stream)
import adapter
importlib.reload(adapter)


class DQNCat(nn.Module):
    def __init__(self, dqn_config, pre_cat, device = "cpu"):
        super().__init__()

        self.input_adapters = []
        self.input_adapters.append(adapter.RnnInputAdapter(dqn_config["rnn"]))
        self.input_adapters.append(adapter.InputAdapter(dqn_config, device))

        streams = []
        streams.append(stream.RnnStream(6, dqn_config["rnn"]))
        streams.append(nn.LazyLinear(2))
        self.streams = nn.ModuleList(streams)
        self.pre_cat = pre_cat
        self.is_factorized = pre_cat.is_factorized

        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"]) if not self.is_factorized else adapter.QValueAdapterMultiDim(dqn_config["categorical"])
        self.action_adapter = adapter.ActionAdapter() if not self.is_factorized else adapter.ActionMultiDimAdapter()
        self.temperature = dqn_config["temperature"]

    def parameters(self):
        params = []
        for stream in self.streams:
            params.extend(stream.parameters())
        #params.extend(self.pre_cat.parameters())
        return params

    def forward(self, x):
        obs = x[:, -5:, :] # [batch_size, sequence_length, obs_space]
        for stream in self.streams:
            x = stream(x)
        x = F.softmax(x / self.temperature, dim = -1) # 温度付きsoftmax
        obs1 = obs[:, :, 2:4]  # [batch_size, sequence_length, 2]
        obs2 = obs[:, :, 4:6]  # [batch_size, sequence_length, 2]
        if self.training:
            # 学習時はattention_weighted_sum
            attention_weighted_sum = x[:, -1:, 0:1] * obs1 + x[:, -1:, 1:2] * obs2  # [batch_size, sequence_length, 2]
            info = attention_weighted_sum
        else:
            # 推論時は確率最大のものを決定論的に選択
            probs = x[:, -1, :]  # [batch_size, 2]
            sampled = torch.argmax(probs, dim=1, keepdim=True)  # [batch_size, 1]
            info = torch.where(sampled == 0, obs1, obs2)  # [batch_size, sequence_length, 2]
        x = torch.cat([obs[:, :, 0:2], info], dim=-1) # [batch_size, sequence_length, 4]
        self.info = info
        x = self.pre_cat(x)
        return x

    def to_input(self, x):
        for adapter in self.input_adapters:
            x = adapter(x)
        return x

    def to_action(self, probabilities):
        return self.pre_cat.to_action(probabilities)

    def get_support(self):
        return self.q_value_adapter.z_support
    
    def reset_noise(self):
        self.pre_cat.reset_noise()
