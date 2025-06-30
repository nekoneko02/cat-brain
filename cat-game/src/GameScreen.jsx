import React, { useEffect, useRef, useState, useCallback } from 'react';
import GameControls from './GameControls';

export default function GameScreen() {
  const gameContainerRef = useRef(null);
  const [gameOver, setGameOver] = useState(false);
  const [isHardMode, setIsHardMode] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [activeDirection, setActiveDirection] = useState(null);

  // ゲームオーバー時に呼ばれるグローバル関数をセット
  useEffect(() => {
    window.onGameOver = () => setGameOver(true);
    return () => { window.onGameOver = undefined; };
  }, []);

  // ゲーム初期化
  const startGame = useCallback(() => {
    setGameOver(false);
    if (window.initializeGame) {
      window.initializeGame();
    }
  }, []);

  // モード・速度・方向をPhaser側に反映
  useEffect(() => {
    if (window.currentGame && window.currentGame.scene && window.currentGame.scene.scenes) {
      const scene = window.currentGame.scene.scenes[0];
      if (scene) {
        scene.isHardMode = isHardMode;
        if (scene.toy && typeof scene.toy.setSpeed === 'function') {
          scene.toy.setSpeed(speed);
        }
        scene.activeDirection = activeDirection;
      }
    }
  }, [isHardMode, speed, activeDirection, gameOver]);

  useEffect(() => {
    startGame();
    return () => {
      if (window.currentGame && window.currentGame.destroy) {
        window.currentGame.destroy(true);
      }
    };
  }, [startGame]);

  // コントロール用ハンドラ
  const handleDirection = dir => setActiveDirection(dir);
  const handleSpeed = s => setSpeed(s);
  const handleMode = mode => setIsHardMode(mode);

  return (
    <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'flex-start', marginTop: 24 }}>
      <div ref={gameContainerRef} id="game-container" style={{ width: 800, height: 600, background: '#eee' }} />
      <div style={{ marginLeft: 32 }}>
        <GameControls
          onDirection={handleDirection}
          onSpeed={handleSpeed}
          onMode={handleMode}
          isHardMode={isHardMode}
          speed={speed}
        />
        {gameOver && (
          <button
            style={{ fontSize: 24, marginTop: 24, padding: '12px 40px', borderRadius: 8, background: '#00bbff', color: '#fff', width: 220 }}
            onClick={startGame}
          >
            リスタート
          </button>
        )}
      </div>
    </div>
  );
}
