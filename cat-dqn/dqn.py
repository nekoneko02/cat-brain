import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.modules as rlnn

class DQN(nn.Module):
    """
    Unified DQN class that supports both RNN and MLP architectures.
    
    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Number of actions
        network_type (str): Type of network architecture ('rnn' or 'mlp')
        num_atoms (int): Number of atoms for distributional RL
        v_min (float): Minimum value for distributional RL
        v_max (float): Maximum value for distributional RL
        hidden_dim (int): Size of hidden layers
        feature_dim (int): Size of feature extraction layers
    """
    def __init__(self, input_dim, output_dim, network_type="mlp", num_atoms=51, 
                 v_min=-10, v_max=10, hidden_dim=64, feature_dim=64):
        super(DQN, self).__init__()
        self.network_type = network_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Distributional RL parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        
        # RNN layer (only used if network_type is 'rnn')
        if network_type == "rnn":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Feature extraction layers
        self.feature = nn.Sequential(
            nn.LazyLinear(feature_dim),
            nn.ReLU(),
            nn.LazyLinear(feature_dim),
            nn.ReLU()
        )
        
        # State value function V(s)
        self.value_stream = nn.Sequential(
            rlnn.NoisyLinear(feature_dim, hidden_dim),
            nn.ReLU(),
            rlnn.NoisyLinear(hidden_dim, num_atoms)
        )
        
        # Advantage function A(s, a)
        self.advantage_stream = nn.Sequential(
            rlnn.NoisyLinear(feature_dim, hidden_dim),
            nn.ReLU(),
            rlnn.NoisyLinear(hidden_dim, output_dim * num_atoms)
        )

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden_state (torch.Tensor, optional): Hidden state for RNN
            
        Returns:
            tuple: (probabilities, hidden_state)
        """
        if self.network_type == "rnn":
            return self._forward_rnn(x, hidden_state)
        else:
            return self._forward_mlp(x)
    
    def _forward_rnn(self, x, hidden_state=None):
        """RNN-specific forward pass"""
        # RNN processing
        x, hidden_state = self.rnn(x, hidden_state)
        x = x[:, -1, :]  # Use only the last output
        
        # Feature extraction
        x = self.feature(x)
        
        # Get probabilities
        probabilities = self._get_probabilities(x)
        return probabilities, hidden_state
    
    def _forward_mlp(self, x):
        """MLP-specific forward pass"""
        # Feature extraction
        x = self.feature(x)
        
        # Get probabilities
        probabilities = self._get_probabilities(x)
        return probabilities, None  # Return None for hidden state to maintain consistent interface
    
    def _get_probabilities(self, x):
        """
        Calculate probabilities using dueling network architecture.
        
        Args:
            x (torch.Tensor): Feature tensor
            
        Returns:
            torch.Tensor: Probability distribution over atoms for each action
        """
        value = self.value_stream(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(-1, self.output_dim, self.num_atoms)
        
        # Distributional Q-values
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_atoms = q_atoms.view(-1, self.output_dim, self.num_atoms)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(q_atoms, dim=2)
        return probabilities
    
    def get_support(self):
        """Get the support for the distributional RL"""
        return self.z_support