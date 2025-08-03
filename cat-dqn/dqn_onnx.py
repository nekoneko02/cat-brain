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
            cat_actions.stop.Stop(),
            cat_actions.escape.Escape()
        ]

    def forward(self, x):
        # optical policy
        option = self.model(x)  # shape: [batch_size] or [batch_size, 1]

        current_obs = x[0, :]  # [observation_space]
        # 各cat_actionの出力をスタック
        action_outputs = []
        for act in self.cat_actions:
            action_outputs.append(act(current_obs).unsqueeze(0))  # shape: [1, action_dim]
        action_outputs = torch.cat(action_outputs, dim=0)  # shape: [num_options, action_dim]

        # option: [batch_size] (index)
        # ONNX対応: gatherで選択（安全な方法）
        option = option.long().view(-1)  # [batch_size]
        batch_size = option.size(0)
        num_options, action_dim = action_outputs.size()
        # action_outputs: [num_options, action_dim] → [1, num_options, action_dim] → [batch_size, num_options, action_dim]
        action_outputs_exp = action_outputs.unsqueeze(0).expand(batch_size, num_options, action_dim)
        # option: [batch_size] → [batch_size, 1, action_dim]
        option_exp = option.view(-1, 1, 1).expand(-1, 1, action_dim)
        selected_cat_action = torch.gather(action_outputs_exp, 1, option_exp).squeeze(1)  # [batch_size, action_dim]
        return option, selected_cat_action