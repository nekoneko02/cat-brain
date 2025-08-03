import importlib
import torch
from gymnasium import spaces

import cat_actions
importlib.reload(cat_actions)

class Cat:
    def __init__(self):
        self.vel = [0.0, 0.0]
        self.energy = 1000
        self.max_energy = 2000
        self.basal_metabolic_rate = 0.25
        self.movement_energy_rate = 0.1

        self.cat_actions = [
            cat_actions.chase.Chase(),
            cat_actions.stop.Stop(),
            cat_actions.escape.Escape()
        ]
    def get_action_space(self):
        num_actions =  len(self.cat_actions)
        return spaces.Discrete(num_actions)

    def set_velocity(self, dx, dy):
        self.vel = [dx, dy]

    def get_action(self, option, obs):
        obs = torch.Tensor(obs)
        action = self.cat_actions[option](obs)
        self.vel = action
        self.energy_consumption(action)
        return {
            "dx": action[0],
            "dy": action[1]
        }

    def eat(self, food_energy):
        # 食べ物のエネルギーを消費
        self.energy += food_energy

        if self.energy > self.max_energy:
            self.energy = self.max_energy
        elif self.energy < 0:
            self.energy = 0

    def get_velocity(self):
        return self.vel

    def get_fatigue(self):
        return max(0.0, self.energy / 1000.0)
    
    def energy_consumption(self, action):
        # PreCatのエネルギー消費を計算
        move_distance = (action[0]**2 + action[1]**2)**0.5
        movement_energy = move_distance * self.movement_energy_rate
        self.energy -= movement_energy
        self.energy -= self.basal_metabolic_rate
        if self.energy < 0:
            self.energy = 0