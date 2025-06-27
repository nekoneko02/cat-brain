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

    def __init__(self, render_mode=None, max_steps=1000, chaser = "cat", runner = "toy", dummy = ["dummy"], reset_interval=1000, n_agents=3):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reset_interval = reset_interval  # 1000stepごとにpossible_agentsを初期化
        self.n_agents = n_agents  # possible_agentsの個数

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

        # possible_agentsの初期化
        self.chaser = chaser
        self.runner = runner
        # dummyをリスト化
        if dummy is None:
            self.dummy = []
        elif isinstance(dummy, list):
            self.dummy = dummy
        else:
            self.dummy = [dummy]
        self.candidates = [self.chaser] + [self.runner] + self.dummy
        self._init_possible_agents()
        self.agents = self.possible_agents[:]
        # CatとToyのサイズを考慮して衝突判定を行う
        self.collision_threshold = {
            agent1: {
                agent2: self._collision_threshold(agent1, agent2) for agent2 in self.candidates
            }
            for agent1 in self.candidates
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
            self.runner: spaces.Discrete(len(self.actions[self.runner]))
        }
        self.action_spaces.update({dummy: spaces.Discrete(len(self.actions[dummy])) for dummy in self.dummy})

        if self.dummy:
            self.dummy_actions = list(range(self.action_spaces[self.dummy[0]].n))
        self.positions = {agent: [0, 0] for agent in self.candidates}
        self._cumulative_rewards = {agent: 0.0 for agent in self.candidates}
        self.rewards = {agent: 0.0 for agent in self.candidates}
        self.terminations = {agent: False for agent in self.candidates}
        self.truncations = {agent: False for agent in self.candidates}
        self.infos = {agent: {} for agent in self.candidates}
        self.all_step_count = 0
        self.grass = np.full((self.width, self.height), 1.0, dtype=np.float64)
        self.glow_grass = 1/(self.width*self.height)
        self.cat_obs_by_toy = [0,0] # toyから見たcatの位置
        self.cat_energy = 1000

    def _collision_threshold(self, agent1, agent2):
        agent1_size = (self.agent_size[agent1]['width'] + self.agent_size[agent1]['height']) / 2
        agent2_size = (self.agent_size[agent2]['width'] + self.agent_size[agent2]['height']) / 2
        return ((agent1_size + agent2_size) / 2) ** 2

    def _sum_grass_in_area(self, center_x, center_y, agent_name=None):
        # agent_nameが指定されていなければrunnerのサイズを使う
        if agent_name is None:
            agent_name = self.runner
        size = self.agent_size[agent_name]
        width = int(size['width'])
        height = int(size['height'])
        x0 = max(int(center_x - width // 2), 0)
        x1 = min(int(center_x + (width + 1) // 2), self.width)
        y0 = max(int(center_y - height // 2), 0)
        y1 = min(int(center_y + (height + 1) // 2), self.height)
        return np.sum(self.grass[x0:x1, y0:y1])

    def observe(self, agent):
        if agent == self.chaser:
            obs = []
            for a in self.possible_agents:
                pos = self.positions[a] 
                obs += pos
            if agent == "cat":
                obs.append(self.cat_energy) # catのエネルギーを追加
            return np.array(obs, dtype=np.float32)

        elif agent == self.runner:
            toy_pos = self.positions[self.runner]
            x, y = int(toy_pos[0]), int(toy_pos[1])
            grass_left  = self._sum_grass_in_area(x-1, y)
            grass_right = self._sum_grass_in_area(x+1, y)
            grass_up    = self._sum_grass_in_area(x, y-1)
            grass_down  = self._sum_grass_in_area(x, y+1)
            return np.array([
                self.cat_obs_by_toy[0], self.cat_obs_by_toy[1],
                toy_pos[0]            , toy_pos[1],
                grass_left,
                grass_right,
                grass_up,
                grass_down
            ], dtype=np.float32)
        elif agent in self.dummy:
            # dummyは使う必要はない
            dummy_pos = self.positions[agent]
            return np.array([dummy_pos[0], dummy_pos[1]], dtype=np.float32)

    def _init_possible_agents(self):
        # 2,3,...番目はrunner/dummyからランダムに選択
        candidates = [self.runner] + self.dummy
        n = max(1, self.n_agents - 1)
        selected = random.sample(candidates, k=n)
        self.possible_agents = [self.chaser] + selected

    def reset_positions(self):
        self._init_possible_agents()
        
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)

        squared_distance = 0
        while squared_distance < 100 ** 2: # 100以上の距離になるように初期値設定
            for agent in self.agents:
                self.positions[agent] = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
                squared_distance = max(
                    squared_distance,
                    self.squared_distance(self.chaser, agent)
                )

    def reset(self, seed=None, options=None):
        self.reset_positions()

        self.agent_selection = self._agent_selector.next()
        self.rewards = {a: 0.0 for a in self.candidates}
        self._cumulative_rewards = {a: 0.0 for a in self.candidates}
        self.terminations = {a: False for a in self.candidates}
        self.truncations = {a: False for a in self.candidates}
        self.infos = {a: {} for a in self.candidates}
        self.all_step_count = 0


        self.cat_obs_by_toy = self.positions[self.chaser]
        self.grass = np.full((self.width, self.height), 1.0, dtype=np.float64)
        self.cat_energy = 1000

        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if agent == self.dummy:
            self._step_dummy()
        elif agent == self.chaser and agent == "pre-cat":
            self._step_pre_cat(action)
        elif agent == self.chaser and agent == "cat":
            self._step_cat(action)
        elif agent == self.runner:
            self._step_runner(action)

        self.all_step_count += 1
        # [dummy, dummy]の場合、reset_interval毎にpossible_agentsを初期化
        if self.runner not in self.possible_agents and self.get_step_count() % self.reset_interval == 0:
            self._init_possible_agents()
            self.agents = self.possible_agents[:]
            self._agent_selector = agent_selector(self.agents)

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
        if self.max_steps and self.get_step_count() >= self.max_steps:
            self.truncations[self.agent] = True
        # dummyはランダムに動く
        action = random.choice(self.dummy_actions)
        self._move_agent(self.dummy, action)
    
    def _step_pre_cat(self, action):
        if self.max_steps and self.get_step_count() >= self.max_steps:
            self.truncations[self.runner] = True
            self.rewards[self.runner] += 100.0 # Toyは生存報酬を得る
            return

        prev_distance = self.squared_distance(self.chaser, self.runner)
        dx, dy = self._move_agent(self.chaser, action)
        selected_action = self.actions[self.chaser][action]

        toy_collision, distance = self._is_collision(self.chaser, self.runner, return_distance = True)

        if selected_action["is_found"] or distance < 100:
            # 動きすぎるか近づきすぎるとtoyに見つかる
            self.cat_obs_by_toy = self.positions[self.chaser]
        
        if toy_collision:
            print("finish by", self.chaser)
            self.rewards[self.chaser] += 200.0
            self.rewards[self.runner] -= 200.0
            self.terminations = {a: True for a in self.agents}
        else:
            self.rewards[self.chaser] += - (abs(dx) + abs(dy))/10 # 動いた分だけ疲労する
            self.rewards[self.chaser] += -0.1 if distance < prev_distance else -1 # 遠ざかると罰. 近づいてもステップ数の罰

    
    def _step_cat(self, action):
        if self.cat_energy <= 0:
            print("cat is tired")
            self.truncations = {agent: True for agent in self.candidates}
            self.rewards[self.chaser] += -100.0
            return
        
        prev_distance = self.squared_distance(self.chaser, self.runner)
        dx, dy = self._move_agent(self.chaser, action)
        selected_action = self.actions[self.chaser][action]

        toy_collision, distance = self._is_collision(self.chaser, self.runner, return_distance = True) if self.runner in self.agents else (False, 0)
        dummy_collision = any(self._is_collision(self.chaser, dummy) for dummy in self.dummy)

        if selected_action["is_found"] or distance < 100:
            # 動きすぎるか近づきすぎるとtoyに見つかる
            self.cat_obs_by_toy = self.positions[self.chaser]
        
        if toy_collision:
            print("finish by", self.chaser)
            self.rewards[self.chaser] += 200.0
            self.rewards[self.runner] -= 200.0
            self.cat_energy += 200
            self.reset_positions()
            return
        
        if dummy_collision:
            self.rewards[self.chaser] += -3.0
        
        self.rewards[self.chaser] += - (abs(dx) + abs(dy))/10 # 動いた分だけ疲労する
        self.rewards[self.chaser] += -0.1 if distance < prev_distance else -1 # 遠ざかると罰. 近づいてもステップ数の罰
        self.cat_energy -= (abs(dx) + abs(dy))/10
        self.cat_energy -= 0.05 # ステップ数の罰

    def _step_runner(self, action):
        if self.max_steps and self.get_step_count() >= self.max_steps:
            self.truncations[self.runner] = True
            self.rewards[self.runner] += 100.0 # Toyは生存報酬を得る
            return
        selected_action = self.actions[self.runner][action]
        self._move_agent(self.runner, action)
        x, y = self.positions[self.runner][0], self.positions[self.runner][1]
        self.rewards[self.runner] += 0.1 # 生存報酬.
        if selected_action["can_eatting"]: # ゆっくり動く時だけ食べられる
            grass_sum = self._sum_grass_in_area(x, y)
            self.rewards[self.runner] += grass_sum # 食べた草の分だけ報酬を得る
            # grassを0にリセット
            size = self.agent_size[self.runner]
            width = int(size['width'])
            height = int(size['height'])
            x0 = max(int(x - width // 2), 0)
            x1 = min(int(x + (width + 1) // 2), self.width)
            y0 = max(int(y - height // 2), 0)
            y1 = min(int(y + (height + 1) // 2), self.height)
            self.grass[x0:x1, y0:y1] = 0
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
            for agent in self.agents:
                agent_x, agent_y = self.positions[agent]
                grid[int(grid_size*(agent_y/self.height))][int(grid_size*(agent_x/self.width))] = agent.upper()  # エージェントの頭文字を大文字で表示
        # (ipynbだけ)ターミナルをクリアするためにclear_outputを使用
        clear_output(wait=True)

        # グリッドを出力（y=0が上になるように反転）
        for row in reversed(grid):
            print(" ".join(row))  # 一行ごとに表示
        print("-" * (2 * grid_size))

        # その他の情報を表示
        formated_positions = ", ".join([f"{agent}: ({self.positions[agent][0]}, {self.positions[agent][1]})" for agent in self.agents])
        print(f"agent: {self.agent_selection}, count: {self.get_step_count()}, positions: {formated_positions}")
        
        # フレーム間の遅延（1フレームごとの更新時間）
        time.sleep(0.01)  # 0.5秒ごとに更新（調整可能）


    def close(self):
        pass