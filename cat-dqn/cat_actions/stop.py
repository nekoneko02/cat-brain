import torch
import adapter

class Stop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        """
        Stop the cat's action in the environment.
        """
        return torch.tensor([0.0, 0.0])  # No movement, cat stays in place