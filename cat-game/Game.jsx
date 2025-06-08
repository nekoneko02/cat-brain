import React, { useEffect, useRef, useState } from 'react';

export default function Game({ onRestart }) {
  const phaserRef = useRef(null);
  const [model, setModel] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);

  // ゲーム初期化関数をuseCallbackで保持
  const startGame = async () => {
    let session = model;
    if (!session) {
      if (!window.ort) throw new Error('ONNX Runtime not loaded');
      session = await window.ort.InferenceSession.create('./cat_dqn_policy.onnx');
      setModel(session);
    }
    setModelLoaded(true);
    if (window.initializeGame) {
      window.initializeGame(session);
    }
  };

  // ゲーム開始時のみ初期化
  useEffect(() => {
    let phaserScript, gameScript, ortScript;
    let isUnmounted = false;
    function injectPhaserAndGame(session) {
      // Phaserを一度だけロード
      if (!window.Phaser || !window.__phaserLoaded) {
        if (!window.__phaserLoading) {
          window.__phaserLoading = true;
          phaserScript = document.createElement('script');
          phaserScript.src = 'https://cdn.jsdelivr.net/npm/phaser@3.55.2/dist/phaser.min.js';
          phaserScript.onload = () => {
            window.__phaserLoaded = true;
            injectGameScript(session);
          };
          document.body.appendChild(phaserScript);
        } else {
          const waitPhaser = () => {
            if (window.__phaserLoaded) {
              injectGameScript(session);
            } else {
              setTimeout(waitPhaser, 50);
            }
          };
          waitPhaser();
        }
      } else {
        injectGameScript(session);
      }
    }
    function injectGameScript(session) {
      if (!window.__catGameScriptLoaded) {
        gameScript = document.createElement('script');
        gameScript.src = './script.js';
        gameScript.type = 'module';
        gameScript.async = true;
        gameScript.onload = () => {
          window.__catGameScriptLoaded = true;
          if (window.initializeGame) {
            window.initializeGame(session);
          }
        };
        document.body.appendChild(gameScript);
      } else {
        if (window.initializeGame) {
          window.initializeGame(session);
        }
      }
    }
    async function loadAndStart() {
      let session = model;
      if (!session) {
        if (!window.ort) throw new Error('ONNX Runtime not loaded');
        session = await window.ort.InferenceSession.create('./cat_dqn_policy.onnx');
        setModel(session);
      }
      setModelLoaded(true);
      if (!window.__catGameStarted) {
        window.__catGameStarted = true;
        injectPhaserAndGame(session);
      }
    }
    // --- ONNX Runtimeのロードを保証 ---
    if (!window.ort) {
      ortScript = document.createElement('script');
      ortScript.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
      ortScript.onload = () => {
        loadAndStart();
      };
      document.body.appendChild(ortScript);
    } else {
      loadAndStart();
    }
    return () => {
      isUnmounted = true;
      if (phaserRef.current) phaserRef.current.innerHTML = '';
      if (window.game && window.game.destroy) {
        window.game.destroy(true, true);
        window.game = null;
      }
      const canvas = document.querySelector('#game-container canvas');
      if (canvas) canvas.remove();
      if (window.Phaser && window.Phaser.Input && window.Phaser.Input.Keyboard && window.Phaser.Input.Keyboard.KeyboardManager) {
        window.Phaser.Input.Keyboard.KeyboardManager._instances = [];
      }
      // ゲーム開始フラグをリセット
      window.__catGameStarted = false;
    };
  }, []);

  // リスタートボタンでGameSceneを破棄し再初期化
  const handleRestart = async () => {
    if (window.game && window.game.destroy) {
      window.game.destroy(true, true);
      window.game = null;
    }
    const canvas = document.querySelector('#game-container canvas');
    if (canvas) canvas.remove();
    if (window.Phaser && window.Phaser.Input && window.Phaser.Input.Keyboard && window.Phaser.Input.Keyboard.KeyboardManager) {
      window.Phaser.Input.Keyboard.KeyboardManager._instances = [];
    }
    setModelLoaded(false);
    // startGame()は呼ばない（initializeGameを再呼び出ししない）
    if (onRestart) onRestart();
  };

  return (
    <div style={{ textAlign: 'center', marginTop: 20 }}>
      <div ref={phaserRef} id="game-container" style={{ margin: '0 auto', width: 800, height: 600, background: '#eee', borderRadius: 12, boxShadow: '0 2px 16px #aaa' }} />
      <button
        style={{ marginTop: 24, fontSize: 22, padding: '10px 40px', background: '#00bbff', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
        onClick={handleRestart}
      >
        リスタート
      </button>
      {!modelLoaded && <div>モデルを読み込み中...</div>}
    </div>
  );
}
