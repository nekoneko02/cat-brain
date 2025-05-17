import torch
import torch.nn as nn

class QValueAdapter(nn.Module):
    def __init__(self, categorical_config):
        self.v_min, self.v_max, self.num_atoms = (
            categorical_config["v_min"],
            categorical_config["v_max"],
            categorical_config["num_atoms"]
        )
        super().__init__()
        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)


    def forward(self, probabilities):
        """
        probabilities: [batch_size, num_actions, num_atoms]
                       or [batch_size, 2, num_actions, num_atoms]
        returns: [batch_size, num_actions]
                 or [batch_size, 2, num_actions]
        """
        q_values = torch.sum(probabilities * self.z_support, dim=-1)
        return q_values
