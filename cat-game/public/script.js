// game_core.jsをESMとして一度だけimport
import { catGameGlobals, linspace, softmax, generateDummyPosition, Cat, Toy, Dummy, GameScene } from './game_core.js';

console.log('[script.js] script.js loaded');

// ゲーム開始処理のみ（グローバル変数やクラス定義はgame_core.jsに依存）

async function loadModel() {
  console.log('[script.js] loadModel called');
  if (catGameGlobals.session) {
    return;
  }
  try {
    console.log('[script.js] Loading model...');
    // public配下のリソースはルート相対で参照
    catGameGlobals.session = await ort.InferenceSession.create('cat_dqn_policy.onnx');
    console.log('[script.js] Model loaded:', catGameGlobals.session);
  } catch (error) {
    console.error('[script.js] Failed to load model:', error);
  }
}

async function loadConfig() {
  console.log('[script.js] loadConfig called');
  const response = await fetch('config.json');
  const data = await response.json();
  catGameGlobals.actions = data.actions.cat;
  catGameGlobals.actions_toy = data.actions.toy;
  catGameGlobals.toy_speed = data.actions.toy_speed_for_game;
  catGameGlobals.observation_space = data.observation_space;
  catGameGlobals.environment = data.environment;
  catGameGlobals.model_config = data.model;
  console.log('[script.js] config loaded', catGameGlobals);
}

// ゲームを初期化する関数
async function initializeGame(session) {
  // catGameGlobalsを初期化（sessionは維持）
  const prevSession = session || catGameGlobals.session;
  Object.keys(catGameGlobals).forEach(k => { if (k !== 'session') delete catGameGlobals[k]; });
  catGameGlobals.session = prevSession;
  console.log('[script.js] initializeGame called', session);
  const container = document.getElementById('game-container');
  if (container) container.innerHTML = '';
  await loadConfig(); // 設定を読み込むまで待機
  if (!catGameGlobals.session) {
    await loadModel();
  }
  const config = {
    type: Phaser.AUTO,
    width: catGameGlobals.environment.width,
    height: catGameGlobals.environment.height,
    parent: 'game-container',
    scene: GameScene,
  };
  console.log('[script.js] Creating Phaser.Game', config);
  // ゲームインスタンスの作成
  window.game = new Phaser.Game(config); // グローバル参照でSPAからも破棄可能に

  // Phaserのcanvasに自動でtabindexとフォーカス
  setTimeout(() => {
    const canvas = document.querySelector('#game-container canvas');
    if (canvas) {
      canvas.setAttribute('tabindex', '0');
      canvas.focus();
    }
  }, 100);
}

// グローバル関数として登録
window.initializeGame = initializeGame;
// 初回自動実行は削除（SPAからのみ呼ぶ）