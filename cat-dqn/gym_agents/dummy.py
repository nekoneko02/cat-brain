import numpy as np
import random
from .runner_base import RunnerBase

class Dummy(RunnerBase):
    def __init__(self, vel_seq_len=1, energy=-1000.0, speed=0.5):
        super().__init__(vel_seq_len=vel_seq_len, energy=energy)
        self.speed = speed

    def get_action(self, observation):
        direction = random.choice([(0, self.speed), (self.speed, 0), (-self.speed, 0), (0, -self.speed)])
        self.vel = np.array([direction[0], direction[1]], dtype=np.float32)
        self.vel_seq.pop(0)
        self.vel_seq.append(self.vel)
        return {"dx": direction[0], "dy": direction[1]}