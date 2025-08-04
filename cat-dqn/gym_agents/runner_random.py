import numpy as np
import random
from .runner_base import RunnerBase

class RunnerRandom(RunnerBase):
    def __init__(self, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = speed

    def get_action(self, observation):
        direction = random.choice([(0, self.speed), (self.speed, 0), (-self.speed, 0), (0, -self.speed)])
        self.vel = np.array([direction[0], direction[1]], dtype=np.float32)
        self.vel_seq.pop(0)
        self.vel_seq.append(self.vel)
        return {"dx": direction[0], "dy": direction[1]}