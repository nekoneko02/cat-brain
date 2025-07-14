class Cat:
    def __init__(self):
        pass
    
    def energy_consumption(self, action):
        # PreCatのエネルギー消費を計算
        move_distance = (action["dx"]**2 + action["dy"]**2)**0.5
        return 0.1 * move_distance # 動いた分だけ疲労する
    
    def basal_metabolic_rate(self):
        # PreCatの基礎代謝を計算
        return 0.05  # 基礎代謝は常に0.05とする