import torch.nn as nn


class FeatureStream(nn.Module):
    def __init__(self, input_dim, layer_configs):
        super().__init__()
        layers = []
        for layer in layer_configs:
            layers.append(nn.LazyLinear(layer))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)