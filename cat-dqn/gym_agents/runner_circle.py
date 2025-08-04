import numpy as np
from .runner_base import RunnerBase

class RunnerCircle(RunnerBase):
    def __init__(self, speed, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle = 0.0  # 円運動の現在角度
        self.radius = radius
        self.speed = speed

    def get_action(self, observation):
        self.angle += self.speed / self.radius
        x = np.cos(self.angle)
        y = np.sin(self.angle)
        direction = np.array([x, y], dtype=np.float32) * self.speed
        self.vel = direction
        self.vel_seq.pop(0)
        self.vel_seq.append(direction)
        return {"dx": direction[0], "dy": direction[1]}