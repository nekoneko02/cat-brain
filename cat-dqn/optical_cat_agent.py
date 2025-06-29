import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
import torch.nn.functional as F
import random
import torch.optim as optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage
from collections import deque

import importlib
import replay_buffer
importlib.reload(replay_buffer)

from replay_buffer import SequenceTensorDictPrioritizedReplayBuffer

class OpticalCatAgent:
    def __init__(self, dqn, target_dqn, cat_dqn, dqn_config, agent_config, device = "cpu", epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.gamma = agent_config["discount_rate"]
        self.device = device

        self.action_space = agent_config["action_space"]

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = dqn
        self.target_model = target_dqn
        self.cat_dqn = cat_dqn
        learning_parameter = self.model.parameters()

        self.optimizer = optim.Adam(learning_parameter, lr=agent_config["learning_rate"])
        self.loss_fn = nn.MSELoss()

        self.has_rnn = "rnn" in dqn_config

        buffer_config = agent_config["buffer"]
        if self.has_rnn:
            self.seq_obs = deque(maxlen=dqn_config["rnn"]["sequence_length"])
            self.memory = SequenceTensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(buffer_config["size"], device = device),
                alpha=buffer_config["alpha"],
                beta=buffer_config["beta"],
                sequence_length=dqn_config["rnn"]["sequence_length"],
            )
        else:
            self.memory = TensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(buffer_config["size"], device = device),
                alpha=buffer_config["alpha"],
                beta=buffer_config["beta"],
            )
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(TensorDict({
            'state': torch.tensor(state, dtype=torch.float32, device=self.device),
            'action': torch.tensor([action], dtype=torch.long, device=self.device),
            'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
            'next_state': torch.tensor(next_state, dtype=torch.float32, device=self.device),
            'done': torch.tensor([done], dtype=torch.float32, device=self.device),
            'td_error': torch.tensor(1.0, dtype=torch.float32, device=self.device)
        }))

    def act(self, state):
        if random.random() <= self.epsilon:
            option = self.action_space.sample()
        else:
            x = state
            x = self.model.to_input(x)
            with torch.no_grad():
                probabilities = self.model.forward(x)
                option = self.model.to_action(probabilities)
        if option == 0:
            x = self.cat_dqn.to_input(state)
            with torch.no_grad():
                probabilities = self.cat_dqn.forward(x)
                action = self.cat_dqn.to_action(probabilities)
            return option, action
        else:
            return option, 8 # stop_action

    def reset_hidden_state(self):
        self.hidden_state = None

    def reset_noise(self):
        self.model.reset_noise()

    def _get_sarsa(self, batch_size, return_info=True):
        batch, info = self.memory.sample(batch_size, return_info=return_info)

        states = batch['state']
        actions = batch['action'].squeeze() 
        rewards = batch['reward'].squeeze()
        next_states = batch['next_state']
        dones = batch['done'].squeeze()

        if self.has_rnn:
            actions = actions[:, -1] # [batch_size, sequence_length]  -> [batch_size] (最後の一つだけ利用する)
            rewards = rewards[:, -1]
            dones = dones[:, -1]
        
        return states, actions, rewards, next_states, dones, info
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        if self.model.is_factorized:
            self._replay_factorized(batch_size)
        else:
            self._replay_simple(batch_size)

    def _replay_simple(self, batch_size):
        states, actions, rewards, next_states, dones, info = self._get_sarsa(batch_size)
        indices, weights = info['index'], info['_weight']
        weights = torch.FloatTensor(weights).to(self.device)  # Tensorに変換

        probabilities = self.model.forward(states)  # [batch_size, num_actions, num_atoms], hidden_state
        with torch.no_grad():
            next_probabilities = self.target_model.forward(next_states)  # [batch_size, num_actions, num_atoms], hidden_state

        batch_indices = torch.arange(batch_size, device=self.device)
        # 選択したアクションの分布を取得
        selected_probs = probabilities[batch_indices, actions] # [batch_size, num_atoms]

        # 次状態の期待Q値の計算
        next_q_values = torch.sum(next_probabilities * self.model.get_support(), dim=-1)  # [batch_size, num_actions]
        next_actions = torch.argmax(next_q_values, dim=1)  # [batch_size]

        # 次状態の分布を選択
        next_dist = next_probabilities[batch_indices, next_actions]

        # Categorical Projection
        projected_distribution = self.project_distribution(rewards, dones, next_dist)

        # 損失計算 (クロスエントロピー損失)
        kl_div = F.kl_div(torch.log(selected_probs + 1e-8), projected_distribution, reduction='none').sum(dim=1)

        # 優先度の更新
        td_errors = kl_div
        # 優先度のクリッピング
        priority = torch.clamp(td_errors.detach(), min=1.0, max=1e3)
        self.memory.update_priority(indices, priority)

        # 損失計算（重み適用）
        loss = (weights * td_errors).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay_factorized(self, batch_size):
        states, actions, rewards, next_states, dones, info = self._get_sarsa(batch_size)
        actions_speed, actions_direction = actions[:, 0], actions[:, 1]
        indices, weights = info['index'], info['_weight']
        weights = torch.FloatTensor(weights).to(self.device)

        # 現在の分布取得
        probabilities_speed, probabilities_direction = self.model.forward(states)

        with torch.no_grad():
            next_probs_speed, next_probs_direction = self.target_model.forward(next_states)

        batch_indices = torch.arange(batch_size, device=self.device)

        # 選択したアクションの分布を取得
        selected_probs_speed = probabilities_speed[batch_indices, actions_speed]  # [batch_size, num_atoms]
        selected_probs_direction = probabilities_direction[batch_indices, actions_direction]

        # 次状態のQ値の計算
        next_q_values_speed = torch.sum(next_probs_speed * self.model.get_support(), dim=-1)
        next_q_values_direction = torch.sum(next_probs_direction * self.model.get_support(), dim=-1)

        # 次状態のアクション選択
        next_actions_speed = torch.argmax(next_q_values_speed, dim=1)
        next_actions_direction = torch.argmax(next_q_values_direction, dim=1)

        # 次状態の分布を選択
        next_dist_speed = next_probs_speed[batch_indices, next_actions_speed]
        next_dist_direction = next_probs_direction[batch_indices, next_actions_direction]

        # プロジェクション
        projected_dist_speed = self.project_distribution(rewards, dones, next_dist_speed)
        projected_dist_direction = self.project_distribution(rewards, dones, next_dist_direction)

        # 損失計算
        kl_div_speed = F.kl_div(torch.log(selected_probs_speed + 1e-8), projected_dist_speed, reduction='none').sum(dim=1)
        kl_div_direction = F.kl_div(torch.log(selected_probs_direction + 1e-8), projected_dist_direction, reduction='none').sum(dim=1)

        # 優先度更新
        td_errors = (kl_div_speed + kl_div_direction)
        priority = torch.clamp(td_errors.detach(), min=1.0, max=1e3)
        self.memory.update_priority(indices, priority)

        # 損失計算（重み適用）
        loss = (weights * td_errors).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def project_distribution(self, rewards, dones, next_dist):
        """
        Categorical Projection for C51 algorithm.

        Args:
            rewards (Tensor): [batch_size] - 報酬
            dones (Tensor): [batch_size] - 終端フラグ
            next_dist (Tensor): [batch_size, num_atoms] - 次状態の分布

        Returns:
            projected_distribution (Tensor): [batch_size, num_atoms] - プロジェクション後の分布
        """
        batch_size = rewards.size(0)
        z_support = self.model.get_support()  # [num_atoms]
        num_atoms = z_support.size(0)
        
        # 各要素の target_z を計算
        target_z = rewards.unsqueeze(1) + self.gamma * z_support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        target_z = target_z.clamp(min=self.model.q_value_adapter.v_min, max=self.model.q_value_adapter.v_max)

        # インデックス計算
        b = (target_z - self.model.q_value_adapter.v_min) / self.model.q_value_adapter.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 下限・上限のクリッピング (無効なインデックスを避けるため)
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)

        offset = torch.arange(batch_size, device=self.device) * num_atoms
        offset = offset.unsqueeze(1)

        # 分布の初期化
        projected_distribution = torch.zeros((batch_size, num_atoms), device=self.device)

        # scatter_add_ による分布更新
        projected_distribution.view(-1).scatter_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )

        projected_distribution.view(-1).scatter_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        return projected_distribution
    def save_model(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.epsilon_min = checkpoint.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = checkpoint.get("epsilon_decay", self.epsilon_decay)
        self.gamma = checkpoint.get("gamma", self.gamma)