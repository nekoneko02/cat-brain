import importlib
import torch
from gymnasium import spaces

import cat_actions
importlib.reload(cat_actions)

class Cat:
    def __init__(self, cat_actions, initial_energy, max_energy, basal_metabolic_rate, movement_energy_rate):
        self.vel = [0.0, 0.0]
        self.energy = initial_energy
        self.max_energy = max_energy
        self.basal_metabolic_rate = basal_metabolic_rate
        self.movement_energy_rate = movement_energy_rate

        self.cat_actions = cat_actions
        
    def get_action_space(self):
        num_actions =  len(self.cat_actions)
        return spaces.Discrete(num_actions)

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