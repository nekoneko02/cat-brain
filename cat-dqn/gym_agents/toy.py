import numpy as np
from .runner_base import RunnerBase

class Toy(RunnerBase):
    def __init__(self, vel_seq_len, energy, speed):
        super().__init__(vel_seq_len=vel_seq_len, energy=energy)
        self.speed = speed

    def get_action(self, observation):
        rel_pos = observation[0:2]
        distance = np.linalg.norm(rel_pos)
        direction = self.speed * rel_pos / (distance + 1e-8)
        self.vel = direction
        self.vel_seq.pop(0)
        self.vel_seq.append(direction)
        return {"dx": direction[0], "dy": direction[1]}
