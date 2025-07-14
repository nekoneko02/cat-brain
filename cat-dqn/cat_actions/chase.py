import torch
import adapter

class Chase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_adapter = adapter.InputAdapter(config=None, device='cpu')
    def forward(self, obs):
        # obs: [cat_x, cat_y, toy_x, toy_y, energy]
        cat_pos = obs[0:2]
        toy_pos = obs[2:4]
        distance_vec = toy_pos - cat_pos
        distance = torch.norm(distance_vec)
        # avoid division by zero
        direction = distance_vec / (distance + 1e-8)
        # dx, dyをテンソルで返す
        return direction  # shape: [2]