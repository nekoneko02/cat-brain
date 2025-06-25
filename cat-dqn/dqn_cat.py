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
        streams.append(stream.RnnStream(7, dqn_config["rnn"]))
        streams.append(stream.FeatureStream(7, dqn_config["feature_stream"]))
        self.feature_stream = nn.ModuleList(streams)

        medium_dim = dqn_config["feature_stream"][-1] + 256
        
        self.dueling_stream = stream.DuelingStream(
            medium_dim,
            dqn_config["value_stream"],
            dqn_config["advantage_stream"],
            dqn_config["categorical"]["num_atoms"]
        )

        self.attention_stream = nn.LazyLinear(2)  # 注意機構(toy or dummyを識別する確率を出力する)
        self.pre_cat = pre_cat
        self.is_factorized = pre_cat.is_factorized

        self.q_value_adapter = adapter.QValueAdapter(dqn_config["categorical"]) if not self.is_factorized else adapter.QValueAdapterMultiDim(dqn_config["categorical"])
        self.action_adapter = adapter.ActionAdapter() if not self.is_factorized else adapter.ActionMultiDimAdapter()
        self.temperature = dqn_config["temperature"]

    def parameters(self):
        params = []
        params.extend(self.feature_stream.parameters())
        params.extend(self.dueling_stream.parameters())
        params.extend(self.attention_stream.parameters())
        #params.extend(self.pre_cat.parameters())
        return params

    def forward(self, x):
        obs = x[:, -5:, :] # [batch_size, sequence_length, obs_space]
        
        # feature_streamの出力を取得
        for stream in self.feature_stream:
            x = stream(x)
        feature = x # [batch_size, 256]
        
        x = self.attention_stream(feature)  # [batch_size, sequence_length, 2]
        x = F.softmax(x / self.temperature, dim = -1) # 温度付きsoftmax
        obs1 = obs[:, :, 2:4]  # [batch_size, sequence_length, 2]
        obs2 = obs[:, :, 4:6]  # [batch_size, sequence_length, 2]
        if self.training:
            # 学習時はattention_weighted_sum
            attention_weighted_sum = x[:, 0:1].unsqueeze(1) * obs1 + x[:, 1:2].unsqueeze(1) * obs2  # [batch_size, sequence_length, 2]
            info = attention_weighted_sum
        else:
            # 推論時は確率最大のものを決定論的に選択
            probs = x  # [batch_size, 2]
            sampled = torch.argmax(probs, dim=1, keepdim=True)  # [batch_size, 1]
            info = torch.where(sampled == 0, obs1, obs2)  # [batch_size, sequence_length, 2]
        x = torch.cat([obs[:, :, 0:2], info], dim=-1) # [batch_size, sequence_length, 4]
        self.info = info
        x = self.pre_cat.streams[0](x)
        x = self.pre_cat.streams[1](x) # [batch_size, 256]
        # feature_streamの出力をdueling_streamに流す
        dueling_out = self.dueling_stream(torch.cat([x, feature], dim=-1) )  # [batch_size, num_atoms]

        return dueling_out

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
