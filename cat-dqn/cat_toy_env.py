import json
import random
import time

import numpy as np
from gymnasium import spaces, Env
from IPython.display import clear_output

COLLISION_THRESHOLD = 3

class CatToyEnv(Env):
    metadata = {"render_modes": ["human"], "name": "cat_toy_env_v0"}

    def __init__(
        self,
        render_mode=None,
        max_steps=1000,
        chaser=None,
        runners=None,
        reset_interval=2000
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reset_interval = reset_interval

        with open("../cat-game/public/common.json") as f:
            config = json.load(f)

        obs_config = config["observation_space"]

        env_config = config["environment"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.max_distance = self.width + self.height
        self.agent_size = env_config["agent_size"]

        self.actions = {
            key: np.array(config["actions"][key]) for key in config["actions"]
        }

        # agent設定
        self.chaser = chaser
        self.runners = runners
        self.candidates = [self.chaser] + self.runners
        # CatとToyのサイズを考慮して衝突判定を行う
        self.collision_threshold = COLLISION_THRESHOLD

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.width - 1, self.height - 1),
            shape=obs_config["cat"]["shape"],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.actions["cat"]))

    def _get_obs(self):
        pos = self.positions[self.chaser] + self.positions[self.current_runner]
        obs = pos + [self.cat_energy]
        return np.array(obs, dtype=np.float32)

    def _init_runner(self):
        self.step_count_from_init_runner = 1
        
        # runnersからランダムに1つ選択
        selected = random.sample(self.runners, k=1)
        self.current_runner = selected[0]
        self.info["current_runner"] = str(self.current_runner)

        while True:
            self.positions[self.current_runner] = [
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            ]
            if not self._is_too_close_agents():
                break
        self.prev_distance = self.squared_distance(self.chaser, self.current_runner)

    def _init_chaser(self):
        while True:
            self.positions[self.chaser] = [
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            ]
            if not self._is_too_close_agents():
                break
        self.prev_distance = self.squared_distance(self.chaser, self.current_runner)

    def _is_too_close_agents(self):
        return self.squared_distance(self.chaser, self.current_runner) < 100**2

    def reset(self, seed=None, options=None):
        self.positions = {agent: [0, 0] for agent in self.candidates}
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {}

        self._init_runner()
        self._init_chaser()

        self.step_count = 0

        self.cat_energy = 1000

        obs = self._get_obs()
        return obs, self.info

    def step(self, action):
        self.step_count += 1
        reward, terminated, truncated, info = self._step_cat(action)            
        self._step_runners()

        # reset_interval毎にrunner再選択
        if self.step_count_from_init_runner >= self.reset_interval:
            self._init_runner()
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _step_cat(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # catの行動
        self._move_agent(self.chaser, action)

        is_collision, distance = self._is_collision(self.chaser, self.current_runner, return_distance=True)
        # 衝突判定
        if is_collision:
            print(f"cat catch toy! {self.info}")
            toy_energy = self.current_runner.get_energy()
            reward += toy_energy
            self.cat_energy += toy_energy
            self._init_runner()

        # エネルギー消費
        energy_consumption = self.chaser.energy_consumption(action)
        basal_metabolic_rate = self.chaser.basal_metabolic_rate()
        reward -= energy_consumption + basal_metabolic_rate
        self.cat_energy -= energy_consumption + basal_metabolic_rate

        # 近寄るボーナス
        if self.current_runner.get_energy() > 0 and distance < self.prev_distance:
            reward += 0.15
        self.prev_distance = distance

        # エネルギー切れ
        if self.cat_energy <= 0:
            print("cat is tired")
            truncated = True
            reward += -10000.0
        return reward, terminated, truncated, info
    def _step_runners(self):
        action = self.current_runner.get_action(self._get_obs())
        self._move_agent(self.current_runner, action)
        
    def _move_agent(self, agent, action):
        dx, dy = action["dx"], action["dy"]
        x, y = self.positions[agent]
        new_x = min(max(x + dx, 0), self.width - 1)
        new_y = min(max(y + dy, 0), self.height - 1)
        self.positions[agent][0] = new_x
        self.positions[agent][1] = new_y

    def squared_distance(self, agent1, agent2):
        agent1_x, agent1_y = self.positions[agent1]
        agent2_x, agent2_y = self.positions[agent2]
        distance = (agent1_x - agent2_x) ** 2 + (agent1_y - agent2_y) ** 2
        return distance

    def _is_collision(self, agent1, agent2, return_distance=False):
        distance = self.squared_distance(agent1, agent2)
        if return_distance:
            return distance < self.collision_threshold, distance
        return distance < self.collision_threshold

    def render(self):
        if self.step_count % 30 != 0:
            return
        grid_size = 30
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        cat_x, cat_y = self.positions[self.chaser]
        toy_x, toy_y = self.positions[self.current_runner]
        if cat_x == toy_x and cat_y == toy_y:
            grid[int(grid_size * (cat_y / self.height))][
                int(grid_size * (cat_x / self.width))
            ] = "C&T"
        else:
            grid[int(grid_size * (cat_y / self.height))][
                int(grid_size * (cat_x / self.width))
            ] = "C"
            grid[int(grid_size * (toy_y / self.height))][
                int(grid_size * (toy_x / self.width))
            ] = "T"
        clear_output(wait=True)
        for row in reversed(grid):
            print(" ".join(row))
        print("-" * (2 * grid_size))
        print(
            f"count: {self.step_count}, positions: cat: ({cat_x}, {cat_y}), toy: ({toy_x}, {toy_y})"
        )
        time.sleep(0.01)

    def close(self):
        pass
