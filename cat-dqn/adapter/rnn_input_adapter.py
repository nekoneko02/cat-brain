import numpy as np
import torch
import torch.nn as nn
from collections import deque


class RnnInputAdapter(nn.Module):
    def __init__(self, rnn_config):
        super().__init__()
        self.seq_obs = deque(maxlen=rnn_config["sequence_length"])

    def forward(self, x):
      self.seq_obs.append(x)
      return self.seq_obs
