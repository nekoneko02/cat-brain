#!/usr/bin/env python3
"""
Test script for Prioritized Experience Replay (PER) implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import PrioritizedReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
import torchrl.modules as rlnn
import numpy as np
import random
from collections import deque
import os
import json
import matplotlib.pyplot as plt
from cat_toy_env import CatToyEnv

# Global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_iterations = 100
num_episodes_per_iteration = 1
num_steps_per_episode = 10000
update_target_steps = 10
replay_interval = 6
buffer_size = 10000
batch_size = 64

# Load configuration
with open('../cat-game/config/common.json', 'r') as f:
    config = json.load(f)
v_max = config["model"]["v_max"]
v_min = config["model"]["v_min"]
num_atoms = config["model"]["num_atoms"]

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_atoms=num_atoms, v_min=v_min, v_max=v_max):
        super(DQN, self).__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.output_dim = output_dim

        # Feature extractor
        self.feature = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 状態価値関数 V(s)
        self.value_stream = nn.Sequential(
            rlnn.NoisyLinear(256, 128),
            nn.ReLU(),
            rlnn.NoisyLinear(128, num_atoms)
        )

        # アドバンテージ関数 A(s, a)
        self.advantage_stream = nn.Sequential(
            rlnn.NoisyLinear(256, 128),
            nn.ReLU(),
            rlnn.NoisyLinear(128, output_dim * num_atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)

        value = self.value_stream(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(-1, self.output_dim, self.num_atoms)

        # Distributional Q-values
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_atoms = q_atoms.view(-1, self.output_dim, self.num_atoms)

        # Apply softmax to get probabilities
        probabilities = F.softmax(q_atoms, dim=2)
        return probabilities # [batch_size, output_dim, num_atoms]

    def get_support(self):
        return self.z_support

# Original DQNAgent (without PER)
class DQNAgent:
    def __init__(self, agent_name, env, learning_rate=1e-4, gamma=0.995, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.agent_name = agent_name  # エージェント名（'cat' または 'toy'）
        self.action_space = env.action_spaces[self.agent_name]  # 各エージェントに対応するアクション空間
        self.state_shape = env.observation_spaces[self.agent_name].shape[0]
        self.model = DQN(self.state_shape, self.action_space.n).to(device)
        self.target_model = DQN(self.state_shape, self.action_space.n).to(device)
        self.state_space = env.observation_spaces[self.agent_name].shape[0]  # 各エージェントに対応する観察空間

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample()  # ランダム行動
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # バッチ次元を追加
        probabilities = self.model(state)  # [batch_size, output_dim, num_atoms]

        # 各アクションごとに期待Q値を計算
        q_values = torch.sum(probabilities * self.model.get_support(), dim=-1)  # [batch_size, output_dim]
        return torch.argmax(q_values).item()  # 最大Q値に基づいて行動を選択

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch = list(zip(*batch))

        states = torch.FloatTensor(np.stack(batch[0])).to(device)
        actions = torch.LongTensor(batch[1]).to(device)
        rewards = torch.FloatTensor(batch[2]).to(device)
        next_states = torch.FloatTensor(np.stack(batch[3])).to(device)
        dones = torch.FloatTensor(batch[4]).to(device)

        # 現在の分布の取得
        probabilities = self.model(states)  # [batch_size, num_actions, num_atoms]
        batch_size, num_actions, num_atoms = probabilities.shape

        # 選択したアクションの分布を取得
        selected_probs = probabilities[torch.arange(batch_size), actions] # [batch_size, num_atoms]

        # 次状態の分布の取得
        next_probabilities = self.target_model(next_states)  # [batch_size, num_actions, num_atoms]

        # 次状態の期待Q値の計算
        next_q_values = torch.sum(next_probabilities * self.model.get_support(), dim=-1)  # [batch_size, num_actions]
        next_actions = torch.argmax(next_q_values, dim=1)  # [batch_size]

        # 次状態の分布を選択
        next_dist = next_probabilities[torch.arange(batch_size), next_actions]

        # Categorical Projection
        projected_distribution = self.project_distribution(rewards, dones, next_dist)
        # 損失計算 (クロスエントロピー損失)
        loss = F.kl_div(torch.log(selected_probs + 1e-8), projected_distribution, reduction='batchmean')

        # パラメータの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # εを減少させる
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
        target_z = target_z.clamp(min=self.model.v_min, max=self.model.v_max)

        # インデックス計算
        b = (target_z - self.model.v_min) / self.model.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 下限・上限のクリッピング (無効なインデックスを避けるため)
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)

        # 分布の割り当て
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).long().unsqueeze(1)

        # 出力分布を初期化
        projected_distribution = torch.zeros((batch_size, num_atoms), device=device)
        
        # 下のインデックスに対して割り当て
        projected_distribution.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )

        # 上のインデックスに対して割り当て
        projected_distribution.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        return projected_distribution

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())

# PER version of DQNAgent
class DQNAgentPER:
    def __init__(self, agent_name, env, learning_rate=1e-4, gamma=0.995, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.agent_name = agent_name  # エージェント名（'cat' または 'toy'）
        self.action_space = env.action_spaces[self.agent_name]  # 各エージェントに対応するアクション空間
        self.state_shape = env.observation_spaces[self.agent_name].shape[0]
        self.model = DQN(self.state_shape, self.action_space.n).to(device)
        self.target_model = DQN(self.state_shape, self.action_space.n).to(device)
        self.state_space = env.observation_spaces[self.agent_name].shape[0]  # 各エージェントに対応する観察空間

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # PrioritizedReplayBufferの初期化
        self.alpha = 0.6  # 優先度の指数
        self.beta = 0.4   # 重要度サンプリングの指数
        self.beta_increment = 0.001  # βの増加量
        self.max_priority = 1.0  # 初期最大優先度
        
        # TorchRLのPrioritizedReplayBufferを使用
        self.memory = PrioritizedReplayBuffer(
            storage=LazyTensorStorage(buffer_size),
            sampler=PrioritizedSampler(
                alpha=self.alpha,
                beta=self.beta,
                max_priority=self.max_priority
            ),
            batch_size=batch_size
        )
        
        self.batch_size = batch_size
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        # PrioritizedReplayBufferに経験を保存
        # 新しい経験には最大優先度を与える
        self.memory.add({
            "state": torch.FloatTensor(state),
            "action": torch.LongTensor([action]) if action is not None else torch.LongTensor([0]),
            "reward": torch.FloatTensor([reward]),
            "next_state": torch.FloatTensor(next_state),
            "done": torch.FloatTensor([float(done)])
        })

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample()  # ランダム行動
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # バッチ次元を追加
        probabilities = self.model(state)  # [batch_size, output_dim, num_atoms]

        # 各アクションごとに期待Q値を計算
        q_values = torch.sum(probabilities * self.model.get_support(), dim=-1)  # [batch_size, output_dim]
        return torch.argmax(q_values).item()  # 最大Q値に基づいて行動を選択

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # βを更新（徐々に1に近づける）
        self.memory.sampler.beta = min(1.0, self.memory.sampler.beta + self.beta_increment)
        
        # PrioritizedReplayBufferからサンプリング
        sample = self.memory.sample()
        
        # サンプルからデータを取得
        states = sample.data["state"].to(device)
        actions = sample.data["action"].squeeze(-1).to(device)
        rewards = sample.data["reward"].squeeze(-1).to(device)
        next_states = sample.data["next_state"].to(device)
        dones = sample.data["done"].squeeze(-1).to(device)
        
        # 重要度サンプリングの重みを取得
        weights = sample.weight.to(device)
        indices = sample.indices
        
        # 現在の分布の取得
        probabilities = self.model(states)  # [batch_size, num_actions, num_atoms]
        batch_size, num_actions, num_atoms = probabilities.shape

        # 選択したアクションの分布を取得
        selected_probs = probabilities[torch.arange(batch_size), actions] # [batch_size, num_atoms]

        # 次状態の分布の取得
        next_probabilities = self.target_model(next_states)  # [batch_size, num_actions, num_atoms]

        # 次状態の期待Q値の計算
        next_q_values = torch.sum(next_probabilities * self.model.get_support(), dim=-1)  # [batch_size, num_actions]
        next_actions = torch.argmax(next_q_values, dim=1)  # [batch_size]

        # 次状態の分布を選択
        next_dist = next_probabilities[torch.arange(batch_size), next_actions]

        # Categorical Projection
        projected_distribution = self.project_distribution(rewards, dones, next_dist)
        
        # 損失計算 (クロスエントロピー損失) - 重要度サンプリングの重みを適用
        elementwise_loss = F.kl_div(
            torch.log(selected_probs + 1e-8), 
            projected_distribution, 
            reduction='none'
        ).sum(dim=1)
        
        # 重み付き損失
        weighted_loss = (elementwise_loss * weights).mean()

        # パラメータの更新
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # TD誤差を優先度として使用
        with torch.no_grad():
            td_errors = elementwise_loss.detach().cpu().numpy()
        
        # 優先度の更新
        priorities = np.abs(td_errors) + 1e-6  # 小さな値を加えて0にならないようにする
        self.memory.update_priorities(indices, priorities)

        # εを減少させる
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
        target_z = target_z.clamp(min=self.model.v_min, max=self.model.v_max)

        # インデックス計算
        b = (target_z - self.model.v_min) / self.model.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 下限・上限のクリッピング (無効なインデックスを避けるため)
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)

        # 分布の割り当て
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).long().unsqueeze(1)

        # 出力分布を初期化
        projected_distribution = torch.zeros((batch_size, num_atoms), device=device)
        
        # 下のインデックスに対して割り当て
        projected_distribution.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )

        # 上のインデックスに対して割り当て
        projected_distribution.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        return projected_distribution

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())