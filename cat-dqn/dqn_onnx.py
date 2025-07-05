import torch
import torch.nn as nn


class DQNOnnx(nn.Module):
    def __init__(self, optical_net, cat_net):
        super().__init__()
        self.model = optical_net
        self.cat_dqn = cat_net

    def forward(self, x):
        # optical policy
        with torch.no_grad():
            probabilities = self.model.forward(x)
            option = self.model.to_action(probabilities)  # option: Tensor[batch_size] or scalar?

        # DQN branch
        with torch.no_grad():
            dqn_prob = self.cat_dqn.forward(x)
            dqn_q = self.cat_dqn.q_value_adapter(dqn_prob)
            dqn_action = self.cat_dqn.action_adapter(dqn_q)
            dqn_info = self.cat_dqn.info  # dummy info (Tensor)

        # Sleep branch
        sleep_action = torch.full_like(dqn_action, 8)  # Tensor型で batch サイズ揃える
        # infoはダミーの値。どこも見ていない=自分自身を見ているというイメージ
        sleep_info = x[:, -5:, 0:2]  # [batch_size, sequence_length, 2] 
        
        # 条件に応じて選ぶ（テンソルベースの分岐）
        option = option.unsqueeze(-1) if option.dim() == 1 else option  # [B, 1]
        action = torch.where(option == 0, dqn_action, sleep_action)
        info = torch.where(option == 0, dqn_info, sleep_info)

        return action, info
