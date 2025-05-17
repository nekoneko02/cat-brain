import numpy as np
import torch
import torch.nn as nn

class InputAdapter(nn.Module):
    def __init__(self, config, device='cpu'):
        super(InputAdapter, self).__init__()
        self.config = config
        self.device = device

    def forward(self, x):
        # x が numpy 配列でない場合の対応
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Tensor に変換してバッチ次元を追加
        return torch.as_tensor(np.array(x)).unsqueeze(0).to(self.device)# バッチ次元を追加
