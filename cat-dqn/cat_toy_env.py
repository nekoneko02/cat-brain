from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np
import json
from IPython.display import clear_output
import random
import os
import time
import pygame

class CatToyEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "cat_toy_env_v0"}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        with open('../cat-game/config/common.json', 'r') as f:
            config = json.load(f)

        env_config = config['environment']
        self.width = env_config['width']
        self.height = env_config['height']
        self.max_distance = self.width + self.height
        self.cat_width = env_config['cat_width']
        self.cat_height = env_config['cat_height']
        self.toy_width = env_config['toy_width']
        self.toy_height = env_config['toy_height']

        self.actions = config['actions']

        self.possible_agents = ['cat', 'dummy', 'toy']
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=max(self.width - 1, self.height - 1), shape=(6,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(len(self.actions[agent]))
            for agent in self.possible_agents
        }

        self.positions = {"cat": (0, 0), "dummy": (0, 0), "toy": (0, 0)}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.step_count = 0

    def observe(self, agent):
        cat_pos = self.positions['cat']
        toy_pos = self.positions['toy']
        dummy_pos = self.positions['dummy']
        return np.array([cat_pos[0], cat_pos[1], toy_pos[0], toy_pos[1], dummy_pos[0], dummy_pos[1]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.step_count = 0

        distance = 0
        while distance < 100: # 100以上の距離になるように初期値設定
            self.positions["cat"] = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            self.positions["toy"] = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            self.positions["dummy"] = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            distance = self.distance("cat", "toy")

        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if agent == "dummy":
            # dummyはランダムに動く
            action = random.choice(range(self.action_spaces["dummy"].n))

        prev_distance = self.distance("cat", "toy")

        selected_action = self.actions[agent][action]
        dx, dy = selected_action["dx"], selected_action["dy"]
        x, y = self.positions[agent]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.positions[agent] = (new_x, new_y)
        else:
            self.rewards[agent] += -1.0  # 壁にぶつかった場合のペナルティ

        collision = self._is_collision()
        if collision:
            print("finish by", agent)
            self.terminations = {a: True for a in self.agents}
            self.rewards["cat"] += 10.0
            self.rewards["toy"] += -10.0
            # ✅ 報酬加算
            self._cumulative_rewards["cat"] += self.rewards["cat"]
            self._cumulative_rewards["toy"] += self.rewards["toy"]
        else:
            distance = self.distance("cat", "toy")
            self.rewards["cat"] += -0.1 if distance < prev_distance else -1 # 遠ざかると罰. 近づいてもステップ数の罰
            self.rewards["toy"] += 0.5 if distance < prev_distance else 1 # 生存報酬. 遠ざかる方がプラス
            # ✅ 報酬加算
            self._cumulative_rewards[agent] += self.rewards[agent]

        if self.step_count >= self.max_steps:
            self.truncations = {a: True for a in self.agents}
        self.step_count += 1

        # ✅ 次のエージェントへ切り替え
        self.agent_selection = self._agent_selector.next()
        self._clear_rewards()
        if self.render_mode == "human":
            self.render()

    def distance(self, agent1, agent2):
        agent1_x, agent1_y = self.positions[agent1]
        agent2_x, agent2_y = self.positions[agent2]
        distance = ((agent1_x - agent2_x) ** 2 + (agent1_y - agent2_y) ** 2) ** 0.5
        return distance
    
    def _is_collision(self):
        distance = self.distance("cat", "toy")
        # CatとToyのサイズを考慮して衝突判定を行う
        cat_size = (self.cat_width + self.cat_height) / 2
        toy_size = (self.toy_width + self.toy_height) / 2
        if distance < (cat_size + toy_size) / 2:
            return True
        
        return False
    
    def render(self):
        if self.step_count % 30 != 0:
            return
        grid_size = 30  # 小さなグリッドに変更
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        cat_x, cat_y = self.positions["cat"]
        toy_x, toy_y = self.positions["toy"]
        # CatとToyの位置に記号を配置（同じ場所なら C&T と表示）
        if cat_x == toy_x and cat_y == toy_y:
            grid[int(grid_size*(cat_y/self.height))][int(grid_size*(cat_x/self.width))] = "C&T"
        else:
            grid[int(grid_size*(cat_y/self.height))][int(grid_size*(cat_x/self.width))] = "C"
            grid[int(grid_size*(toy_y/self.width))][int(grid_size*(toy_x/self.width))] = "T"

        # (ipynbだけ)ターミナルをクリアするためにclear_outputを使用
        clear_output(wait=True)

        # グリッドを出力（y=0が上になるように反転）
        for row in reversed(grid):
            print(" ".join(row))  # 一行ごとに表示
        print("-" * (2 * grid_size))

        # その他の情報を表示
        cat_x, cat_y = self.positions["cat"]
        toy_x, toy_y = self.positions["toy"]
        print(f"agent: {self.agent_selection}, count: {self.step_count}, cat: {cat_x}, {cat_y}, toy: {toy_x}, {toy_y}")

        # フレーム間の遅延（1フレームごとの更新時間）
        time.sleep(0.05)  # 0.5秒ごとに更新（調整可能）


    def close(self):
        pass