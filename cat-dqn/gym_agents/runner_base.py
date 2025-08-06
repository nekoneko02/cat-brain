import numpy as np

class RunnerBase:
    def __init__(self, vel_seq_len, energy, collision_threshold=3, escape_distance =1000, escape_steps=2000):
        self.vel_seq_len = vel_seq_len
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.vel_seq = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(vel_seq_len)]
        self.energy = energy
        self.collision_threshold = collision_threshold
        self.escape_distance  = escape_distance 
        self.escape_steps = escape_steps

        self.step_count = 0

    def get_velocity(self):
        return np.array(self.vel_seq[::-1]).flatten()

    def get_energy(self):
        return self.energy

    def get_action(self, obs):
        self.step_count += 1
        return self._get_action(obs)

    def _get_action(self, obs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def is_reset(self, cat_pos, toy_pos):
        return self.is_caught(cat_pos, toy_pos) or self.is_escaped(cat_pos, toy_pos)

    def is_caught(self, cat_pos, toy_pos):
        dx = cat_pos[0] - toy_pos[0]
        dy = cat_pos[1] - toy_pos[1]
        distance = dx ** 2 + dy ** 2
        return distance < self.collision_threshold

    def is_escaped(self, cat_pos, toy_pos):
        return self.is_too_long(cat_pos, toy_pos) or self.is_too_step_count()
    
    def is_too_long(self, cat_pos, toy_pos):
        # 設定がない場合、常に存在する
        if self.escape_distance  is None:
            return False
        dx = cat_pos[0] - toy_pos[0]
        dy = cat_pos[1] - toy_pos[1]
        distance = dx ** 2 + dy ** 2
        return distance > self.escape_distance ** 2

    def is_too_step_count(self):
        return self.step_count >= self.escape_steps