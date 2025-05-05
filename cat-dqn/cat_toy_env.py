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
        self.cat_width = env_config['cat_width']
        self.cat_height = env_config['cat_height']
        self.toy_width = env_config['toy_width']
        self.toy_height = env_config['toy_height']
        self.state_scale = env_config['state_scale']

        self.actions = config['actions']

        self.possible_agents = ['cat', 'toy']
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(1, int(self.height*self.state_scale), int(self.width*self.state_scale)), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            "cat": spaces.Discrete(len(self.actions["cat"])),
            "toy": spaces.Discrete(len(self.actions["toy"])),
        }

        # ✅ PettingZoo AECEnv に必要な属性
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.step_count = 0
        
    def observe(self, agent):
        # 1/10にスケールして、stateを返す
        scaled_cat_x = int(self.cat_x * self.state_scale)
        scaled_cat_y = int(self.cat_y * self.state_scale)
        scaled_toy_x = int(self.toy_x * self.state_scale)
        scaled_toy_y = int(self.toy_y * self.state_scale)

        obs = np.zeros(self.observation_spaces["cat"].shape, dtype=np.float32)

        # Cat の位置（矩形領域）を 1 に設定
        obs[0,
            scaled_cat_y:scaled_cat_y + int(self.cat_height*self.state_scale),
            scaled_cat_x:scaled_cat_x + int(self.cat_width*self.state_scale)
        ] = 1.0
        # Toy の位置（矩形領域）を 0.5 に設定（オーバーラップ判定のため）
        obs[0,
            scaled_toy_y:scaled_toy_y + int(self.toy_height*self.state_scale),
            scaled_toy_x:scaled_toy_x + int(self.toy_width*self.state_scale)
        ] = 0.5

        return obs


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.dones = {a: False for a in self.agents}  # 互換性のために残してもOK
        self.infos = {a: {} for a in self.agents}
        self.step_count = 0

        self.cat_x = random.randint(0, self.width - self.cat_width)
        self.cat_y = random.randint(0, self.height - self.cat_height)
        self.toy_x = random.randint(0, self.width - self.toy_width)
        self.toy_y = random.randint(0, self.height - self.toy_height)

        return self.observe(self.agent_selection)


    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        selected_action = self.actions[agent][action]
        dx = selected_action["dx"]
        dy = selected_action["dy"]

        if agent == "cat":
            self.cat_x = np.clip(self.cat_x + dx, 0, self.width - self.cat_width)
            self.cat_y = np.clip(self.cat_y + dy, 0, self.height - self.cat_height)
        elif agent == "toy":
            self.toy_x = np.clip(self.toy_x + dx, 0, self.width - self.toy_width)
            self.toy_y = np.clip(self.toy_y + dy, 0, self.height - self.toy_height)

        collision = self._is_collision()
        if collision:
            self.terminations = {a: True for a in self.agents}
            self.rewards["cat"] = 100.0
            self.rewards["toy"] = -100.0
        elif self.step_count >= self.max_steps:
            self.truncations = {a: True for a in self.agents}
        else:        
            distance = ((self.cat_x - self.toy_x) ** 2 + (self.cat_y - self.toy_y) ** 2) ** 0.5
            self.rewards["cat"] = -distance
            self.rewards["toy"] = distance
        if self.step_count >= self.max_steps:
            self.truncations[agent] = True

        # ✅ 報酬加算
        self._cumulative_rewards[agent] += self.rewards[agent]
        self.step_count += 1

        # ✅ 次のエージェントへ切り替え
        self.agent_selection = self._agent_selector.next()
        
        if self.render_mode == "human":
            self.render()

    def _is_collision(self):
        return (
            self.cat_x < self.toy_x + self.toy_width and
            self.cat_x + self.cat_width > self.toy_x and
            self.cat_y < self.toy_y + self.toy_height and
            self.cat_y + self.cat_height > self.toy_y
        )

    def render(self):
        grid_size = 30  # 小さなグリッドに変更
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        # CatとToyの位置に記号を配置（同じ場所なら C&T と表示）
        if self.cat_x == self.toy_x and self.cat_y == self.toy_y:
            grid[self.cat_y*grid_size//self.height][self.cat_x*grid_size//self.width] = "C&T"
        else:
            grid[self.cat_y*grid_size//self.height][self.cat_x*grid_size//self.width] = "C"
            grid[self.toy_y*grid_size//self.height][self.toy_x*grid_size//self.width] = "T"

        # (ipynbだけ)ターミナルをクリアするためにclear_outputを使用
        clear_output(wait=True)

        # グリッドを出力（y=0が上になるように反転）
        for row in reversed(grid):
            print(" ".join(row))  # 一行ごとに表示
        print("-" * (2 * grid_size))

        # その他の情報を表示
        print(f"agent: {self.agent_selection}, count: {self.step_count}, cat: {self.cat_x}, {self.cat_y}, toy: {self.toy_x}, {self.toy_y}")

        # フレーム間の遅延（1フレームごとの更新時間）
        time.sleep(0.05)  # 0.5秒ごとに更新（調整可能）


    def close(self):
        pass  # 特にリソース解放がなければ空でOK