
import torch.nn as nn
import torch.nn.functional as F

from .categorical_stream import CategoricalStream


class DuelingStream(nn.Module):
    def __init__(self, input_dim, value_stream_config, advantage_stream_config, num_atoms):
        super().__init__()
        self.value_stream = CategoricalStream(input_dim, value_stream_config, num_atoms)
        self.advantage_stream = CategoricalStream(input_dim, advantage_stream_config, num_atoms)

    def forward(self, x):
        value_output = self.value_stream(x)
        advantage_output = self.advantage_stream(x)
        q_atoms = value_output + advantage_output - advantage_output.mean(dim=1, keepdim=True)
        probabilities = F.softmax(q_atoms, dim=2)
        return probabilities
    
    def reset_noise(self):
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()