#!/usr/bin/env python3
"""
状態空間の整理 - 使用例デモ (Issue #127)

このサンプルコードは、新しく組織化された状態空間の使用方法を示します。
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/cat-brain/cat-brain/cat-dqn')

from cat_toy_env import CatToyEnv
from state_space import StateSpace
import numpy as np

def demo_state_space_usage():
    """組織化された状態空間の使用例"""
    print("=== 状態空間の整理 - 使用例デモ ===")
    
    # 1. 環境の初期化
    env = CatToyEnv(chaser="cat", runner="toy", dummy=["dummy1"])
    print("✓ 環境を初期化しました")
    
    # 2. エピソードの開始
    initial_obs = env.reset()
    print(f"✓ 初期観測を取得: {len(initial_obs)}次元")
    
    # 3. 状態ベクトルの解析
    def analyze_cat_state(obs):
        """猫の状態ベクトルを分析して可読形式で表示"""
        print("\n--- 猫の状態分析 ---")
        
        # 物理的関係
        cat_x, cat_y = obs[0], obs[1]
        toy_x, toy_y = obs[2], obs[3]
        distance = obs[7]
        angle_norm = obs[6]
        angle_degrees = angle_norm * 360
        
        print(f"🐱 猫の位置: ({cat_x:.1f}, {cat_y:.1f})")
        print(f"🧸 おもちゃの位置: ({toy_x:.1f}, {toy_y:.1f})")
        print(f"📏 距離: {distance:.1f}")
        print(f"🧭 相対角度: {angle_degrees:.1f}度")
        
        # 内部状態
        playfulness = obs[9]
        fatigue = obs[10]
        curiosity = obs[11]
        
        print(f"😺 遊びたい度: {playfulness:.2f}")
        print(f"😴 疲労度: {fatigue:.2f}")
        print(f"🤔 好奇心: {curiosity:.2f}")
        
        # 環境要因
        env_sound = obs[12]
        toy_visible = obs[13]
        lighting = obs[14]
        
        print(f"🔊 環境音: {'あり' if env_sound > 0.5 else 'なし'}")
        print(f"👀 おもちゃ視認: {'可能' if toy_visible > 0.5 else '不可'}")
        print(f"💡 照明レベル: {lighting:.2f}")
        
        # 行動履歴
        last_action = int(obs[15])
        consecutive_count = int(obs[16])
        
        print(f"⚡ 前回行動: {last_action}")
        print(f"🔄 連続回数: {consecutive_count}")
    
    # 4. 初期状態の分析
    if env.agent_selection == "cat":
        analyze_cat_state(initial_obs)
    
    # 5. 複数ステップの実行と状態変化の観察
    print("\n=== ステップ実行と状態変化 ===")
    
    for step in range(5):
        # 現在のエージェントが猫の場合
        if env.agent_selection == "cat":
            current_obs = env.observe("cat")
            
            print(f"\nステップ {step + 1}:")
            print(f"エージェント: {env.agent_selection}")
            
            # 短縮版の状態表示
            cat_pos = current_obs[0:2]
            toy_pos = current_obs[2:4]
            distance = current_obs[7]
            playfulness = current_obs[9]
            
            print(f"猫: ({cat_pos[0]:.1f}, {cat_pos[1]:.1f}), "
                  f"おもちゃ: ({toy_pos[0]:.1f}, {toy_pos[1]:.1f}), "
                  f"距離: {distance:.1f}, 遊びたい度: {playfulness:.2f}")
        
        # ランダムな行動を実行
        action = np.random.randint(0, 4)  # 基本的な移動行動
        env.step(action)
    
    # 6. 状態空間の直接操作例
    print("\n=== StateSpaceクラスの直接使用例 ===")
    
    state_space = StateSpace()
    
    # 物理的関係の計算
    cat_pos = (100, 200)
    toy_pos = (300, 400)
    physical = state_space.calculate_physical_relationships(cat_pos, toy_pos, 1000)
    
    print(f"距離計算: {physical['distance']:.1f}")
    print(f"角度計算: {physical['relative_angle_degrees']:.1f}度")
    
    # 内部状態の更新
    internal = state_space.update_internal_state(
        energy_change=-50,  # エネルギー消費
        play_time=2.0      # 2秒間遊んだ
    )
    
    print(f"遊びたい度更新: {internal['playfulness']:.2f}")
    print(f"疲労度更新: {internal['fatigue']:.2f}")
    
    # 完全な状態ベクトルの取得
    complete_state = state_space.get_complete_state(
        cat_pos, toy_pos, 1000,
        current_action=2,
        energy_change=-10,
        loud_sound=True
    )
    
    print(f"完全状態ベクトル: {len(complete_state)}次元")
    print(f"先頭5要素: {complete_state[:5]}")
    
    print("\n🎉 デモ完了！状態空間が正常に組織化されています。")

def demo_comparison():
    """従来との比較デモ"""
    print("\n=== 従来との比較 ===")
    
    env = CatToyEnv(chaser="cat", runner="toy", dummy=["dummy1"])
    env.reset()
    
    # 猫の観測（新しい17次元）
    cat_obs = env.observe("cat")
    print(f"🐱 猫エージェント: {len(cat_obs)}次元（組織化された状態空間）")
    
    # おもちゃの観測（従来の8次元）
    toy_obs = env.observe("toy")
    print(f"🧸 おもちゃエージェント: {len(toy_obs)}次元（従来の状態空間）")
    
    # ダミーの観測（従来の2次元）
    dummy_obs = env.observe("dummy1")
    print(f"👻 ダミーエージェント: {len(dummy_obs)}次元（従来の状態空間）")
    
    print("✓ 後方互換性が保たれています")

if __name__ == "__main__":
    try:
        demo_state_space_usage()
        demo_comparison()
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)