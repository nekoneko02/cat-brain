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

    def __init__(self, render_mode=None, max_steps=1000, chaser = "cat", runner = "toy", dummy = "dummy"):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        with open('../cat-game/config/common.json', 'r') as f:
            config = json.load(f)
        
        obs_config = config['observation_space']

        env_config = config['environment']
        self.width = env_config['width']
        self.height = env_config['height']
        self.max_distance = self.width + self.height
        self.agent_size = env_config['agent_size']

        self.actions = {
            key: np.array(config['actions'][key]) 
                for key in config['actions']
        }

        self.chaser = chaser
        self.runner = runner
        self.dummy = dummy
        self.possible_agents = [chaser, runner, dummy]
        self.possible_agents = [agent for agent in self.possible_agents if agent is not None]
        self.agents = self.possible_agents[:]
        # CatとToyのサイズを考慮して衝突判定を行う
        self.collision_threshold = {
            agent1: {
                agent2: self._collision_threshold(agent1, agent2) for agent2 in self.agents
            }
            for agent1 in self.agents
        }
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=max(self.width - 1, self.height - 1), shape=obs_config[agent]["shape"], dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            "cat": spaces.Discrete(len(self.actions["cat"])),
            "pre-cat": spaces.Discrete(len(self.actions["pre-cat"])),
            self.runner: spaces.Discrete(len(self.actions[self.runner])),
            self.dummy: spaces.Discrete(len(self.actions[self.dummy])) if self.dummy else None
        }

        if self.dummy:
            self.dummy_actions = list(range(self.action_spaces[self.dummy].n))
        self.positions = {agent: [0, 0] for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.all_step_count = 0
        self.grass = np.full((self.width, self.height), 1.0, dtype=np.float64)
        self.glow_grass = 1/(self.width*self.height)
        self.cat_obs_by_toy = [0,0] # toyから見たcatの位置

    def _collision_threshold(self, agent1, agent2):
        agent1_size = (self.agent_size[agent1]['width'] + self.agent_size[agent1]['height']) / 2
        agent2_size = (self.agent_size[agent2]['width'] + self.agent_size[agent2]['height']) / 2
        return ((agent1_size + agent2_size) / 2) ** 2

    def observe(self, agent):
        cat_pos = self.positions[self.chaser]
        toy_pos = self.positions[self.runner]

        if agent == self.chaser and self.dummy:
            dummy_pos = self.positions[self.dummy]
            return np.array([
                self.positions[self.possible_agents[0]][0], self.positions[self.possible_agents[0]][1],
                self.positions[self.possible_agents[1]][0], self.positions[self.possible_agents[1]][1],
                self.positions[self.possible_agents[2]][0], self.positions[self.possible_agents[2]][1]
            ], dtype=np.float32)
        elif agent == self.chaser:
            return np.array([cat_pos[0], cat_pos[1], toy_pos[0], toy_pos[1]], dtype=np.float32)
        elif agent == self.runner:
            x, y = int(toy_pos[0]), int(toy_pos[1])
            return np.array([
                self.cat_obs_by_toy[0], self.cat_obs_by_toy[1],
                toy_pos[0]            , toy_pos[1],
                self.grass[x - 1, y] if x-1 >= 0          else 0,
                self.grass[x + 1, y] if x+1 < self.width  else 0,
                self.grass[x, y - 1] if y-1 >= 0          else 0,
                self.grass[x, y + 1] if y+1 < self.height else 0
            ], dtype=np.float32)
        elif agent == self.dummy:
            # dummyは使う必要はない
            dummy_pos = self.positions[self.dummy]
            return np.array([dummy_pos[0], dummy_pos[1]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if self.dummy:
            cat = self.possible_agents[0]
            toy = self.possible_agents[1]
            dummy = self.possible_agents[2]
            # ランダムにAgentsの順番を設定する
            if random.random() > 0.5:
                self.possible_agents[0] = cat
                self.possible_agents[1] = toy
                self.possible_agents[2] = dummy
            else:
                self.possible_agents[0] = cat
                self.possible_agents[1] = dummy
                self.possible_agents[2] = toy

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.all_step_count = 0

        squared_distance = 0
        while squared_distance < 100 ** 2: # 100以上の距離になるように初期値設定
            for agent in self.agents:
                self.positions[agent] = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            squared_distance = self.squared_distance(self.chaser, self.runner)

        self.cat_obs_by_toy = self.positions[self.chaser]
        self.grass = np.full((self.width, self.height), 1.0, dtype=np.float64)

        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if agent == self.dummy:
            self._step_dummy()
        elif agent == self.chaser:
            self._step_chaser(action)
        elif agent == self.runner:
            self._step_runner(action)

        if self.get_step_count() >= self.max_steps:
            self.truncations[agent] = {agent: True for agent in self.agents}
            self.rewards[self.runner] += 100.0 # Toyは生存報酬を得る
            self.rewards[self.chaser] += -100.0 # Catは罰を受ける
        self.all_step_count += 1

        # ✅ 報酬加算
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]


        # ✅ 次のエージェントへ切り替え
        self.agent_selection = self._agent_selector.next()
        self._clear_rewards()

        if self.render_mode == "human":
            self.render()
    def get_step_count(self):
        return self.all_step_count // len(self.possible_agents)

    def _step_dummy(self):
        # dummyはランダムに動く
        action = random.choice(self.dummy_actions)
        self._move_agent(self.dummy, action)
    
    def _step_chaser(self, action):
        prev_distance = self.squared_distance(self.chaser, self.runner)
        dx, dy = self._move_agent(self.chaser, action)
        selected_action = self.actions[self.chaser][action]

        toy_collision, distance = self._is_collision(self.chaser, self.runner, return_distance = True)
        dummy_collision = self.dummy and self._is_collision(self.chaser, self.dummy)

        if selected_action["is_found"] or distance < 100:
            # 動きすぎるか近づきすぎるとtoyに見つかる
            self.cat_obs_by_toy = self.positions[self.chaser]
        
        if toy_collision:
            print("finish by", self.chaser)
            self.terminations = {a: True for a in self.agents}
            self.rewards[self.chaser] += 200.0
            self.rewards[self.runner] -= 200.0
        elif dummy_collision:
            print("dummy finish by", self.chaser)
            self.terminations = {a: True for a in self.agents}
            self.rewards[self.chaser] += -300.0
        else:
            self.rewards[self.chaser] += - (abs(dx) + abs(dy))/10 # 動いた分だけ疲労する
            self.rewards[self.chaser] += -0.1 if distance < prev_distance else -1 # 遠ざかると罰. 近づいてもステップ数の罰

    def _step_runner(self, action):
        selected_action = self.actions[self.runner][action]
        self._move_agent(self.runner, action)
        x, y = self.positions[self.runner][0], self.positions[self.runner][1]
        self.rewards[self.runner] += 0.1 # 生存報酬.
        if selected_action["can_eatting"]: # ゆっくり動く時だけ食べられる
            self.rewards[self.runner] += self.grass[int(x), int(y)] # 食べた草の分だけ報酬を得る
            self.grass[int(x), int(y)] = 0
        self.grass += self.glow_grass

    def _move_agent(self, agent, action):
        selected_action = self.actions[agent][action]
        speed, dx, dy = selected_action["speed"], selected_action["dx"], selected_action["dy"]
        dx, dy = dx * speed, dy * speed
        
        x, y = self.positions[agent]
        new_x, new_y = x + dx, y + dy

        # 画面外に出ないようにする
        new_x = min(max(x + dx, 0), self.width - 1)
        new_y = min(max(y + dy, 0), self.height - 1)

        self.positions[agent][0] = new_x
        self.positions[agent][1] = new_y

        return dx, dy

    def squared_distance(self, agent1, agent2):
        agent1_x, agent1_y = self.positions[agent1]
        agent2_x, agent2_y = self.positions[agent2]
        distance = (agent1_x - agent2_x) ** 2 + (agent1_y - agent2_y) ** 2
        return distance
    
    def _is_collision(self, agent1, agent2, return_distance = False):
        distance = self.squared_distance(agent1, agent2)

        if distance < self.collision_threshold[agent1][agent2]:
            result = True
        else:
            result = False

        if return_distance:
            return result, distance
        return result
    
    def render(self):
        if self.get_step_count() % 30 != 0:
            return
        grid_size = 30  # 小さなグリッドに変更
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        cat_x, cat_y = self.positions[self.chaser]
        toy_x, toy_y = self.positions[self.runner]
        # CatとToyの位置に記号を配置（同じ場所なら C&T と表示）
        if cat_x == toy_x and cat_y == toy_y:
            grid[int(grid_size*(cat_y/self.height))][int(grid_size*(cat_x/self.width))] = "C&T"
        else:
            grid[int(grid_size*(cat_y/self.height))][int(grid_size*(cat_x/self.width))] = "C"
            grid[int(grid_size*(toy_y/self.height))][int(grid_size*(toy_x/self.width))] = "T"
            if self.dummy:
                dum_x, dum_y = self.positions[self.dummy]
                grid[int(grid_size*(dum_y/self.height))][int(grid_size*(dum_x/self.width))] = "D"

        # (ipynbだけ)ターミナルをクリアするためにclear_outputを使用
        clear_output(wait=True)

        # グリッドを出力（y=0が上になるように反転）
        for row in reversed(grid):
            print(" ".join(row))  # 一行ごとに表示
        print("-" * (2 * grid_size))

        # その他の情報を表示
        cat_x, cat_y = self.positions[self.chaser]
        toy_x, toy_y = self.positions[self.runner]
        if self.dummy:
            print(f"agent: {self.agent_selection}, count: {self.get_step_count()}, cat: {cat_x}, {cat_y}, toy: {toy_x}, {toy_y}, dummy: {dum_x}, {dum_y}")
        else:
            print(f"agent: {self.agent_selection}, count: {self.get_step_count()}, cat: {cat_x}, {cat_y}, toy: {toy_x}, {toy_y}")
        
        # フレーム間の遅延（1フレームごとの更新時間）
        time.sleep(0.01)  # 0.5秒ごとに更新（調整可能）


    def close(self):
        pass