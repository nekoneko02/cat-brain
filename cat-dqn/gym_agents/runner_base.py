import numpy as np

class RunnerBase:
    def __init__(self, vel_seq_len, energy):
        self.vel_seq_len = vel_seq_len
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.vel_seq = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(vel_seq_len)]
        self.energy = energy

    def get_velocity(self):
        return np.array(self.vel_seq[::-1]).flatten()

    def get_energy(self):
        return self.energy
