import importlib

import stream
import torch.nn as nn

importlib.reload(stream)


class DQNFactory:
    def __init__(self, dqn_config):
        self.dqn_config = dqn_config

    def create_streams(self):
        streams = []
        input_dim = self.dqn_config["input_dim"]

        if "rnn" in self.dqn_config:
            streams.append(stream.RnnStream(input_dim, self.dqn_config["rnn"]))

        streams.append(stream.FeatureStream(input_dim, self.dqn_config["feature"]))
        medium_dim = self.dqn_config["feature"][-1]

        is_factorized = "speed_advantage_stream" in self.dqn_config and "direction_advantage_stream" in self.dqn_config

        if is_factorized:
            streams.append(stream.FactorizedStream(medium_dim, self.dqn_config["value_stream"], self.dqn_config["speed_advantage_stream"], self.dqn_config["direction_advantage_stream"], self.dqn_config["categorical"]["num_atoms"]))
        else:
            streams.append(stream.DuelingStream(medium_dim, self.dqn_config["value_stream"], self.dqn_config["advantage_stream"], self.dqn_config["categorical"]["num_atoms"]))

        return nn.ModuleList(streams)