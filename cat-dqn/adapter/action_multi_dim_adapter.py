
import torch
import torch.nn as nn
class ActionMultiDimAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values):
        """
        q_values: [batch_size, num_actions]
                  or [batch_size, 2, num_actions]
        returns: [batch_size]
                 or [batch_size, 2]
        """
        actions = []
        for q_value in q_values:
            actions.append(torch.argmax(q_value, dim=-1))
        return actions
