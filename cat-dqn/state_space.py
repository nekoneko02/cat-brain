"""
状態空間の整理：強化学習で使用する状態を組織化したクラス

Issues #127で定義された状態空間の構成要素を実装:
1. ねことおもちゃの物理的関係
2. ねこの内部状態  
3. 環境ノイズ・外部要因
4. 行動履歴
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque


class StateSpace:
    """状態空間を組織化するクラス"""
    
    def __init__(self, max_history_length: int = 10):
        """
        状態空間を初期化
        
        Args:
            max_history_length: 行動履歴の最大長
        """
        self.max_history_length = max_history_length
        self.action_history = deque(maxlen=max_history_length)
        self.consecutive_action_count = 0
        self.last_action = None
        
        # 内部状態の初期化
        self.playfulness = 1.0  # 遊びたい度 (0.0-1.0)
        self.fatigue = 0.0  # 疲労度 (0.0-1.0)
        self.curiosity = 0.8  # 好奇心パラメータ (性格)
        
        # 環境要因の初期化
        self.environmental_sound = 0  # 環境音 (0/1)
        self.toy_visible = True  # おもちゃが視界に入っているか
        
    def calculate_physical_relationships(
        self, 
        cat_pos: Tuple[float, float], 
        toy_pos: Tuple[float, float],
        max_distance: float
    ) -> Dict[str, float]:
        """
        ねことおもちゃの物理的関係を計算
        
        Args:
            cat_pos: ねこの位置 (x, y)
            toy_pos: おもちゃの位置 (x, y)
            max_distance: 環境の最大距離（正規化用）
            
        Returns:
            物理的関係の辞書
        """
        cat_x, cat_y = cat_pos
        toy_x, toy_y = toy_pos
        
        # ユークリッド距離の計算
        distance = math.sqrt((cat_x - toy_x) ** 2 + (cat_y - toy_y) ** 2)
        
        # 正規化された距離 (0.0-1.0)
        normalized_distance = min(distance / max_distance, 1.0)
        
        # 近接レベル (距離の逆数的な指標)
        proximity_level = 1.0 - normalized_distance
        
        # 相対角度の計算 (ねこから見たおもちゃの方向)
        dx = toy_x - cat_x
        dy = toy_y - cat_y
        
        # atan2で角度を計算（-π〜πを0〜2πに変換）
        angle_rad = math.atan2(dy, dx)
        if angle_rad < 0:
            angle_rad += 2 * math.pi
            
        # 角度を0-360度に変換
        angle_degrees = math.degrees(angle_rad)
        
        # 正規化された角度 (0.0-1.0)
        normalized_angle = angle_degrees / 360.0
        
        return {
            'cat_x': cat_x,
            'cat_y': cat_y,
            'toy_x': toy_x,
            'toy_y': toy_y,
            'distance': distance,
            'normalized_distance': normalized_distance,
            'proximity_level': proximity_level,
            'relative_angle_degrees': angle_degrees,
            'normalized_relative_angle': normalized_angle
        }
        
    def update_internal_state(
        self, 
        energy_change: float = 0.0,
        toy_appeared: bool = False,
        rest_time: float = 0.0,
        play_time: float = 0.0
    ) -> Dict[str, float]:
        """
        ねこの内部状態を更新
        
        Args:
            energy_change: エネルギーの変化量
            toy_appeared: 新しいおもちゃが登場したか
            rest_time: 休憩時間
            play_time: 遊び時間
            
        Returns:
            内部状態の辞書
        """
        # 遊びたい度の更新
        if toy_appeared:
            self.playfulness = min(self.playfulness + 0.3, 1.0)
        if rest_time > 0:
            self.playfulness = min(self.playfulness + rest_time * 0.1, 1.0)
        if play_time > 0:
            self.playfulness = max(self.playfulness - play_time * 0.05, 0.0)
            
        # 疲労度の更新（エネルギーの逆指標）
        if energy_change < 0:
            self.fatigue = min(self.fatigue - energy_change * 0.001, 1.0)
        if rest_time > 0:
            self.fatigue = max(self.fatigue - rest_time * 0.1, 0.0)
            
        # 疲労度が高いと遊びたい度も下がる
        if self.fatigue > 0.7:
            self.playfulness = max(self.playfulness - 0.1, 0.0)
            
        return {
            'playfulness': self.playfulness,
            'fatigue': self.fatigue,
            'curiosity': self.curiosity
        }
        
    def update_environmental_factors(
        self,
        loud_sound: bool = False,
        obstacles_present: bool = False,
        lighting_level: float = 1.0
    ) -> Dict[str, float]:
        """
        環境ノイズ・外部要因を更新
        
        Args:
            loud_sound: 大きな音があったか
            obstacles_present: 障害物があるか
            lighting_level: 照明レベル (0.0-1.0)
            
        Returns:
            環境要因の辞書
        """
        self.environmental_sound = 1 if loud_sound else 0
        
        # おもちゃの視界判定（照明と障害物を考慮）
        self.toy_visible = lighting_level > 0.3 and not obstacles_present
        
        return {
            'environmental_sound': float(self.environmental_sound),
            'toy_visible': float(self.toy_visible),
            'lighting_level': lighting_level
        }
        
    def update_action_history(self, action: int) -> Dict[str, Any]:
        """
        行動履歴を更新
        
        Args:
            action: 実行された行動のID
            
        Returns:
            行動履歴の辞書
        """
        # 行動履歴に追加
        self.action_history.append(action)
        
        # 連続行動カウントの更新
        if self.last_action == action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
            
        self.last_action = action
        
        # one-hotエンコード（簡略版：最大10行動と仮定）
        max_actions = 10
        action_one_hot = [0] * max_actions
        if action < max_actions:
            action_one_hot[action] = 1
            
        # 履歴配列（固定長）
        history_array = list(self.action_history) + [0] * (self.max_history_length - len(self.action_history))
        
        return {
            'last_action': action,
            'action_one_hot': action_one_hot,
            'consecutive_action_count': self.consecutive_action_count,
            'action_history': history_array
        }
        
    def get_complete_state(
        self,
        cat_pos: Tuple[float, float],
        toy_pos: Tuple[float, float],
        max_distance: float,
        current_action: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        完全な状態ベクトルを取得
        
        Args:
            cat_pos: ねこの位置
            toy_pos: おもちゃの位置
            max_distance: 環境の最大距離
            current_action: 現在の行動
            **kwargs: その他のパラメータ
            
        Returns:
            組織化された状態ベクトル
        """
        # 1. 物理的関係
        physical = self.calculate_physical_relationships(cat_pos, toy_pos, max_distance)
        
        # 2. 内部状態の更新
        internal = self.update_internal_state(
            energy_change=kwargs.get('energy_change', 0.0),
            toy_appeared=kwargs.get('toy_appeared', False),
            rest_time=kwargs.get('rest_time', 0.0),
            play_time=kwargs.get('play_time', 0.0)
        )
        
        # 3. 環境要因の更新
        environmental = self.update_environmental_factors(
            loud_sound=kwargs.get('loud_sound', False),
            obstacles_present=kwargs.get('obstacles_present', False),
            lighting_level=kwargs.get('lighting_level', 1.0)
        )
        
        # 4. 行動履歴の更新
        if current_action is not None:
            action_info = self.update_action_history(current_action)
        else:
            action_info = {
                'last_action': self.last_action or 0,
                'consecutive_action_count': self.consecutive_action_count
            }
        
        # 状態ベクトルの構築
        state_vector = [
            # 物理的関係 (9要素)
            physical['cat_x'], physical['cat_y'],
            physical['toy_x'], physical['toy_y'],
            physical['normalized_distance'],
            physical['proximity_level'],
            physical['normalized_relative_angle'],
            physical['distance'],  # 元の距離も保持
            physical['relative_angle_degrees'] / 360.0,  # 正規化角度
            
            # 内部状態 (3要素)
            internal['playfulness'],
            internal['fatigue'],
            internal['curiosity'],
            
            # 環境要因 (3要素)
            environmental['environmental_sound'],
            float(environmental['toy_visible']),
            environmental['lighting_level'],
            
            # 行動履歴 (2要素)
            float(action_info['last_action']),
            float(action_info['consecutive_action_count'])
        ]
        
        return np.array(state_vector, dtype=np.float32)
        
    def get_state_dimension(self) -> int:
        """状態ベクトルの次元数を取得"""
        return 17  # 9 + 3 + 3 + 2 = 17次元
        
    def reset(self):
        """状態空間をリセット"""
        self.action_history.clear()
        self.consecutive_action_count = 0
        self.last_action = None
        self.playfulness = 1.0
        self.fatigue = 0.0
        self.environmental_sound = 0
        self.toy_visible = True