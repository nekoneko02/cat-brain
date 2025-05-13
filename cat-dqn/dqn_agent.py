import torch
import torch.nn as nn
from tensordict import TensorDict
import torch.nn.functional as F
import random
import torch.optim as optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage
from collections import deque

import importlib
import dqn
import replay_buffer
importlib.reload(dqn)
importlib.reload(replay_buffer)

from dqn import DQN
from replay_buffer import SequenceTensorDictPrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, dqn_config, agent_config, device = "cpu", epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.gamma = agent_config["discount_rate"]
        self.device = device

        
        self.action_space = agent_config["action_space"]

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN(dqn_config).to(device)
        self.target_model = DQN(dqn_config).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=agent_config["learning_rate"])
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
            'state': torch.FloatTensor(state),
            'action': torch.LongTensor([action]),
            'reward': torch.FloatTensor([reward]),
            'next_state': torch.FloatTensor(next_state),
            'done': torch.FloatTensor([done]),
            'td_error': 1.0 # 初期の誤差は1に設定
        }))

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample()
        if self.has_rnn:
            self.seq_obs.append(state)
            input = self.seq_obs
        else:
            input = state
        input = torch.FloatTensor(input).unsqueeze(0).to(self.device)# バッチ次元を追加
        probabilities = self.model(input)  # [batch_size, output_dim, num_atoms]
        # 各アクションごとに期待Q値を計算
        q_values = torch.sum(probabilities * self.model.get_support(), dim=-1)  # [batch_size, output_dim]
        return torch.argmax(q_values).item()  # 最大Q値に基づいて行動を選択

    def reset_hidden_state(self):
        self.hidden_state = None

    def _get_sarsa(self, batch_size, return_info=True):
        batch, info = self.memory.sample(batch_size, return_info=return_info)

        states = batch['state']
        actions = batch['action'].squeeze() 
        rewards = batch['reward'].squeeze()
        next_states = batch['next_state']
        dones = batch['done'].squeeze()
        if self.has_rnn:
            actions = actions[:, -1]                    # [batch_size, sequence_length]  -> [batch_size] (最後の一つだけ利用する)
            rewards = rewards[:, -1]
            dones = dones[:, -1]
        return states, actions, rewards, next_states, dones, info
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones, info = self._get_sarsa(batch_size)
        indices, weights = info['index'], info['_weight']
        weights = torch.FloatTensor(weights).to(self.device)  # Tensorに変換

        # 現在の分布の取得
        probabilities = self.model(states)  # [batch_size, num_actions, num_atoms], hidden_state
        batch_size, num_actions, num_atoms = probabilities.shape

        batch_indices = torch.arange(batch_size, device=self.device)
        # 選択したアクションの分布を取得
        selected_probs = probabilities[batch_indices, actions] # [batch_size, num_atoms]

        # 次状態の分布の取得
        next_probabilities = self.target_model(next_states)  # [batch_size, num_actions, num_atoms], hidden_state

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
        td_errors = kl_div.detach()
        # 優先度のクリッピング
        max_priority = 1e3  # 適宜調整
        td_errors = torch.clamp(td_errors, min=1.0, max=max_priority)
        self.memory.update_priority(indices, td_errors)

        # 損失計算（重み適用）
        loss = (weights * kl_div).mean()
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
        target_z = target_z.clamp(min=self.model.v_min, max=self.model.v_max)

        # インデックス計算
        b = (target_z - self.model.v_min) / self.model.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 下限・上限のクリッピング (無効なインデックスを避けるため)
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)

        # 分布の割り当て
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=self.device).long().unsqueeze(1)

        # 出力分布を初期化
        projected_distribution = torch.zeros((batch_size, num_atoms), device=self.device)
        
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
