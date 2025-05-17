
import torch
import torch.nn as nn
class ActionAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values):
        """
        q_values: [batch_size, num_actions]
                  or [batch_size, 2, num_actions]
        returns: [batch_size]
                 or [batch_size, 2]
        """
        actions = torch.argmax(q_values, dim=-1)
        return actions
