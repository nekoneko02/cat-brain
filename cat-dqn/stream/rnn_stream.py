import torch.nn as nn


class RnnStream(nn.Module):
  def __init__(self, input_dim, rnn_config):
      super().__init__()
      hidden_dim = rnn_config["hidden_dim"]
      self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
      self.return_sequence = rnn_config["return_sequence"] if "return_sequence" in rnn_config else False

  def forward(self, x):
      x, _ = self.rnn(x, None)
      if self.return_sequence:
          return x[:, -5:, :]
      else:
          return x[:, -1, :]
