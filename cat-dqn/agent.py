import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torchrl.data import TensorDictPrioritizedReplayBuffer, PrioritizedSampler, LazyTensorStorage
from tensordict import TensorDict

from dqn import DQN

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceTensorDictPrioritizedReplayBuffer(TensorDictPrioritizedReplayBuffer):
    """
    Extended TensorDictPrioritizedReplayBuffer that supports sequence sampling.
    
    Args:
        storage: Storage backend for the buffer
        max_capacity (int): Maximum capacity of the buffer
        alpha (float): Priority exponent
        beta (float): Importance sampling exponent
        sequence_length (int): Length of sequences to sample
    """
    def __init__(self, storage, max_capacity, alpha=0.6, beta=0.4, sequence_length=5):
        sampler = PrioritizedSampler(max_capacity=max_capacity, alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        super().__init__(storage=storage, alpha=alpha, beta=beta)
        self.sequence_length = sequence_length

    def sample(self, batch_size, sequence_length=None, return_info=True):
        """
        Sample sequences from the buffer.
        
        Args:
            batch_size (int): Number of sequences to sample
            sequence_length (int, optional): Length of sequences
            return_info (bool): Whether to return sampling info
            
        Returns:
            tuple: (batch_data, info)
        """
        sequence_length = sequence_length or self.sequence_length

        # Get indices based on priority
        indices, info = self._sampler.sample(self._storage, batch_size)
        info['index'] = indices

        # Calculate sequence start indices
tuple: (batch_data, info)
        """
        sequence_length = sequence_length or self.sequence_length
        
        # Ensure sequence_length doesn't exceed buffer size
        max_sequence_length = min(sequence_length, len(self._storage))

        # Get indices based on priority
        indices, info = self._sampler.sample(self._storage, batch_size)
        info['index'] = indices

        # Calculate sequence start indices
        start_indices = indices - (max_sequence_length - 1)
        start_indices = start_indices.clamp(min=0)

        # Collect sequences
        indices = torch.arange(max_sequence_length).unsqueeze(0) + start_indices.unsqueeze(1)  # [batch_size, max_sequence_length]

        batch_data = self._storage.get(indices.flatten())
        batch_data = batch_data.view(batch_size, max_sequence_length, *batch_data.shape[1:])
        # Transfer batch data to device
        batch_data = batch_data.to(device)
        start_indices = start_indices.clamp(min=0)

        # Collect sequences
        indices = torch.arange(sequence_length).unsqueeze(0) + start_indices.unsqueeze(1)  # [batch_size, sequence_length]

        batch_data = self._storage.get(indices.flatten())
        batch_data = batch_data.view(batch_size, sequence_length, *batch_data.shape[1:])
        # Transfer batch data to device
        batch_data = batch_data.to(device)
    
        return batch_data, info

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices (torch.Tensor): Indices to update
            td_errors (torch.Tensor): TD errors for priority calculation
        """
        td_errors = td_errors.view(len(indices), -1).max(dim=1)[0]
        # Update priorities directly
        self._sampler.update_priorities(indices, td_errors.detach().cpu().numpy())

class DQNAgent:
    """
    Unified DQN Agent that supports both RNN and MLP models.
    
    Args:
        agent_name (str): Name of the agent
        env: Environment the agent interacts with
        network_type (str): Type of network ('rnn' or 'mlp')
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for exploration
        hidden_dim (int): Size of hidden layers
        feature_dim (int): Size of feature extraction layers
        num_atoms (int): Number of atoms for distributional RL
        v_min (float): Minimum value for distributional RL
        v_max (float): Maximum value for distributional RL
        buffer_size (int): Size of replay buffer
        batch_size (int): Batch size for training
        sequence_length (int): Length of sequences for RNN (ignored for MLP)
    """
    def __init__(self, agent_name, env, network_type="mlp", learning_rate=1e-4, 
                 gamma=0.995, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 hidden_dim=64, feature_dim=64, num_atoms=51, v_min=-10, v_max=10,
                 buffer_size=10000, batch_size=64, sequence_length=5):
        self.agent_name = agent_name
        self.network_type = network_type
        
        # Learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Environment information
        self.action_space = env.action_spaces[self.agent_name]
        self.state_shape = env.observation_spaces[self.agent_name].shape[0]
        
        # Create appropriate replay buffer based on network type
        if network_type == "rnn":
            self.memory = SequenceTensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(buffer_size),
                max_capacity=buffer_size,
                alpha=0.6,
                beta=0.4,
                sequence_length=sequence_length
            )
            self.hidden_state = None
            self.seq_obs = deque(maxlen=sequence_length)
        else:
            self.memory = TensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(buffer_size),
                alpha=0.6,
                beta=0.4,
            )
        
        # Create model and target model
        self.model = DQN(
            input_dim=self.state_shape,
            output_dim=self.action_space.n,
            network_type=network_type,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim
        ).to(device)
        
        self.target_model = DQN(
            input_dim=self.state_shape,
            output_dim=self.action_space.n,
            network_type=network_type,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """Update target model with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.add(TensorDict({
            'state': torch.FloatTensor(state),
            'action': torch.LongTensor([action]),
            'reward': torch.FloatTensor([reward]),
            'next_state': torch.FloatTensor(next_state),
            'done': torch.FloatTensor([done]),
            'td_error': torch.tensor(1.0)  # Initial error set to 1
        }))

    def act(self, state):
        """
        Select action based on current policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if random.random() <= self.epsilon:
            return self.action_space.sample()
        
        probabilities = self._call_model_for_act(state)
        # Calculate expected Q values
        q_values = torch.sum(probabilities * self.model.get_support(), dim=-1)
        return torch.argmax(q_values).item()

    def reset_hidden_state(self):
        """Reset hidden state for RNN"""
        self.hidden_state = None
        
    def _call_model_for_act(self, state):
        """
        Call model for action selection.
        
        Args:
            state: Current state
            
        Returns:
            torch.Tensor: Probability distribution
        """
        if self.network_type == "rnn":
            self.seq_obs.append(state)
            state_tensor = torch.FloatTensor(list(self.seq_obs)).unsqueeze(0).to(device)
            probabilities, self.hidden_state = self.model(state_tensor, self.hidden_state)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probabilities, _ = self.model(state_tensor)
            
        return probabilities

    def _call_model(self, state):
        """
        Call model for training.
        
        Args:
            state: Batch of states
            
        Returns:
            torch.Tensor: Probability distribution
        """
        probabilities, _ = self.model(state)
        return probabilities
        
    def _call_target_model(self, state):
        """
        Call target model for training.
        
        Args:
            state: Batch of states
            
        Returns:
            torch.Tensor: Probability distribution
        """
        probabilities, _ = self.target_model(state)
        return probabilities

    def _get_sarsa(self, batch_size, return_info=True):
        """
        Sample experiences from replay buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            return_info (bool): Whether to return sampling info
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, info)
        """
        batch, info = self.memory.sample(batch_size, return_info=return_info)
        
        states = batch['state'].to(device)
        actions = batch['action'].to(device).squeeze()
        rewards = batch['reward'].to(device).squeeze()
        next_states = batch['next_state'].to(device)
        dones = batch['done'].to(device).squeeze()
        
        # For RNN, we need to use the last step in the sequence
        if self.network_type == "rnn":
            actions = actions[:, -1]
            rewards = rewards[:, -1]
            dones = dones[:, -1]
            
        return states, actions, rewards, next_states, dones, info

    def replay(self):
        """Train the agent by replaying experiences"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, info = self._get_sarsa(self.batch_size)
        indices, weights = info['index'], info['_weight']
        weights = torch.FloatTensor(weights).to(device)

        # Get current distribution
        probabilities = self._call_model(states)
        batch_size, num_actions, num_atoms = probabilities.shape

        batch_indices = torch.arange(batch_size, device=device)
        # Get distribution for selected actions
        selected_probs = probabilities[batch_indices, actions]

        # Get next state distribution
        next_probabilities = self._call_target_model(next_states)

        # Calculate expected Q values for next state
        next_q_values = torch.sum(next_probabilities * self.model.get_support(), dim=-1)
        next_actions = torch.argmax(next_q_values, dim=1)

        # Get distribution for next actions
        next_dist = next_probabilities[batch_indices, next_actions]

        # Categorical projection
        projected_distribution = self.project_distribution(rewards, dones, next_dist)

        # Calculate loss (KL divergence)
        kl_div = F.kl_div(torch.log(selected_probs + 1e-8), projected_distribution, reduction='none').sum(dim=1)

        # Update priorities
        td_errors = kl_div.detach()
        max_priority = 1e3
        td_errors = torch.clamp(td_errors, min=1.0, max=max_priority)
        self.memory.update_priorities(indices, td_errors)

        # Calculate weighted loss
        loss = (weights * kl_div).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def project_distribution(self, rewards, dones, next_dist):
        """
        Categorical projection for C51 algorithm.
        
        Args:
            rewards (torch.Tensor): Batch of rewards
            dones (torch.Tensor): Batch of done flags
            next_dist (torch.Tensor): Next state distribution
            
        Returns:
            torch.Tensor: Projected distribution
        """
        batch_size = rewards.size(0)
        z_support = self.model.get_support()
        num_atoms = z_support.size(0)
        
        # Calculate target_z
        target_z = rewards.unsqueeze(1) + self.gamma * z_support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        target_z = target_z.clamp(min=self.model.v_min, max=self.model.v_max)

        # Calculate indices
        b = (target_z - self.model.v_min) / self.model.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Clip indices
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)

        # Calculate offset
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).long().unsqueeze(1)

        # Initialize output distribution
        projected_distribution = torch.zeros((batch_size, num_atoms), device=device)
        
        # Assign to lower index
        projected_distribution.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )

        # Assign to upper index
        projected_distribution.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        return projected_distribution

    def save_model(self, filepath):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model weights from file"""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())