import torch.nn as nn

class DQNOnnx(nn.Module):
    def __init__(self, dqn):
        super().__init__()

        self.dqn = dqn

    def forward(self, x):
        x = self.dqn(x)
        q_values = self.dqn.q_value_adapter(x)
        action = self.dqn.action_adapter(q_values)
        return action, q_values