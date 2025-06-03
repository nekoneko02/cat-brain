import torch.nn as nn
import torchrl.modules as rlnn

class CategoricalStream(nn.Module):
    def __init__(self, input_dim, layer_configs, num_atoms):
        super().__init__()
        self.num_atoms = num_atoms
        layers = []
        for layer in layer_configs[:-1]:
            layers.append(rlnn.NoisyLinear(input_dim, layer))
            layers.append(nn.ReLU())
            input_dim = layer
        layers.append(rlnn.NoisyLinear(input_dim, layer_configs[-1] * num_atoms))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.network(x).view(batch_size, -1, self.num_atoms)
        return output
    
    def reset_noise(self):
        for layer in self.network:
            if isinstance(layer, rlnn.NoisyLinear):
                layer.reset_noise()