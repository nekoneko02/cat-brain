import numpy as np
import torch.nn as nn


class RnnStream(nn.Module):
  def __init__(self, input_dim, rnn_config):
      super().__init__()
      hidden_dim = rnn_config["hidden_dim"]
      self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
  def forward(self, x):
      x, _ = self.rnn(x, None)
      return x[:, -1, :]  # 最後の出力のみを使用