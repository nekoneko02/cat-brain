import torch
import adapter

class Escape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_adapter = adapter.InputAdapter(config=None, device='cpu')
    def forward(self, obs):
        # obs: [rel_x, rel_y, chaser_vel_x, chaser_vel_y, runner_vel_x, runner_vel_y, fatigue]
        rel_pos = obs[0:2]
        
        distance = torch.norm(rel_pos)
        direction = rel_pos / (distance + 1e-8)
        return -direction  # shape: [2]