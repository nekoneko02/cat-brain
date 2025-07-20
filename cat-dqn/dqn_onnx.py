import torch
import torch.nn as nn
import importlib

import cat_actions
from tensordict import TensorDict
importlib.reload(cat_actions)

class DQNOnnx(nn.Module):
    def __init__(self, optical_net):
        super().__init__()
        self.model = optical_net

        self.cat_actions = [
            cat_actions.chase.Chase(),
            cat_actions.stop.Stop()
        ]

    def forward(self, x):
        # optical policy
        td = self.model.actor(TensorDict({"observation": x}, batch_size=x.shape[:1]))
        option = td["action"]

        current_obs = x[0, -1, :] # [observation_space]
        selected_cat_action = torch.where(option == 0, self.cat_actions[0](current_obs), self.cat_actions[1](current_obs))  # 0: chase, 1: stop
        return option, selected_cat_action