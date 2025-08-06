from .runner_base import RunnerBase
import numpy as np

class RunnerOscillation(RunnerBase):
    def __init__(self, axis, amplitude, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        self.amplitude = amplitude
        self.freq = freq
        self.t = 0

    def _get_action(self, obs):
        self.t += 1
        if self.axis == 'x':
            dx = self.amplitude * np.sin(self.t * self.freq)
            dy = 0.0
        else:
            dx = 0.0
            dy = self.amplitude * np.sin(self.t * self.freq)
        return {'dx': float(dx), 'dy': float(dy)}
