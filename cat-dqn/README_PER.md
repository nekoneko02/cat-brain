# Prioritized Experience Replay (PER) Implementation

This directory contains an implementation of Prioritized Experience Replay (PER) for the DQN agent in the cat-toy environment.

## Files

- `dqn_agent_per.py`: Contains the DQNAgent class with PER implementation
- `dqn_agent_per_implementation.py`: Contains a standalone implementation of the DQNAgent with PER

## How to Use

### Option 1: Use the standalone implementation

1. Import the DQNAgent class from `dqn_agent_per_implementation.py`
2. Create an instance of the DQNAgent class
3. Use the agent in your training loop

### Option 2: Modify the existing notebook

To modify the existing `cat-dqn.ipynb` notebook to use PER:

1. Replace the `store_experience()` method in the DQNAgent class:

```python
def __init__(self, agent_name, env, learning_rate=1e-4, gamma=0.995, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    # ... existing code ...
    
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
    
    # ... existing code ...

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
```

2. Replace the `replay()` method in the DQNAgent class:

```python
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
```

## Performance Comparison

To check the performance of the PER implementation:

1. Run the original DQN implementation and record the results
2. Run the PER implementation and record the results
3. Compare the learning curves, final performance, and training stability

PER should generally provide:
- Faster learning
- Better final performance
- More stable training

## References

- [Prioritized Experience Replay paper](https://arxiv.org/abs/1511.05952)
- [TorchRL documentation](https://pytorch.org/rl/)