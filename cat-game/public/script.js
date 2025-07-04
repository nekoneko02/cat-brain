let debugMode = false; // デバッグモードフラグ
// let session;

async function loadModel() {
  try {
    let base = window.base_path || '/';
    if (!base.endsWith('/')) base += '/';
    console.log('Loading model...');
    session = await ort.InferenceSession.create(base+'cat_dqn_policy.onnx');
    console.log('Model loaded:', session);
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}


let actions = [];
let actions_toy = [];
let toy_speed = [];
let observation_space = {};
let environment = {};
let model_config = {};
let hidden_state = null;

async function loadConfig() {
  let base = window.base_path || '/';
  if (!base.endsWith('/')) base += '/';
  const response = await fetch(base + 'common.json');
  const data = await response.json();
  actions = data.actions.cat;
  actions_toy = data.actions.toy;
  toy_speed = data.actions.toy_speed_for_game;
  observation_space = data.observation_space;
  environment = data.environment;
  model_config = data.model;
}

function setGameConfigFromGlobal() {
  if (!window.catGameConfig) {
    throw new Error('window.catGameConfig が未定義です。React側でセットしてください。');
  }
  const data = window.catGameConfig;
  actions = data.actions.cat;
  actions_toy = data.actions.toy;
  toy_speed = data.actions.toy_speed_for_game;
  observation_space = data.observation_space;
  environment = data.environment;
  model_config = data.model;
}

function linspace(v_min, v_max, num_atoms) {
  const arr = new Array(num_atoms);
  for (let i = 0; i < num_atoms; i++) {
      arr[i] = v_min + (v_max - v_min) * (i / (num_atoms - 1));
  }
  return arr;
}

function softmax(arr, temperature = 1.0) {
  // 温度パラメータで分布の鋭さを調整
  const maxVal = Math.max(...arr);
  const expArr = arr.map(v => Math.exp((v - maxVal) / temperature));
  const sumExp = expArr.reduce((sum, val) => sum + val, 0);
  return expArr.map(v => v / sumExp);
}

class Cat extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, init_input, scale) {
    super(scene, x, y, 'cat');
    this.setScale(scale);
    this.seq_obs = []
    for(let seq_i=0; seq_i < model_config.sequence_length; seq_i++){
      this.seq_obs[seq_i] = init_input;
    }
    this.interest = [];
    this.dummyPosition = [init_input[4], init_input[5]];
    // info可視化用サークル
    this.infoCircle = scene.add.circle(0, 0, 10, 0xffff00, 0.7);
    this.infoCircle.setVisible(false);
  }

  async move(toy, dummy) {
    const {action, info} = await this.predictAction(this, toy, dummy);
    const selectedAction = actions[action];
    if (selectedAction) {
      this.x += selectedAction.dx * selectedAction.speed;
      this.y += selectedAction.dy * selectedAction.speed;
    }

    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);

    // infoの可視化
    if (debugMode && info && info.length >= 2) {
      // infoが[2*seq]の1次元配列の場合、seq個の点を描画
      let seq = info.length / 2;
      if (Number.isInteger(seq) && seq > 1) {
        // 既存のinfoCircleを削除
        if (this.infoCircles) {
          this.infoCircles.forEach(c => c.destroy());
        }
        this.infoCircles = [];
        for (let i = 0; i < seq; i++) {
          const x = info[i * 2];
          const y = info[i * 2 + 1];
          if (typeof x === 'number' && typeof y === 'number') {
            const circle = this.scene.add.circle(x, y, 8, 0xffff00, 0.7);
            this.infoCircles.push(circle);
          }
        }
      } else {
        // 1点のみの場合
        if (!this.infoCircle) {
          this.infoCircle = this.scene.add.circle(0, 0, 10, 0xffff00, 0.7);
        }
        this.infoCircle.setPosition(info[0], info[1]);
        this.infoCircle.setVisible(true);
      }
    } else {
      // 非表示
      if (this.infoCircles) {
        this.infoCircles.forEach(c => c.destroy());
        this.infoCircles = [];
      }
      if (this.infoCircle) {
        this.infoCircle.setVisible(false);
      }
    }
  }
  async predictAction(cat, toy, dummy) {
    if (!session) throw new Error('Model not loaded yet!');

    const input = [
      cat.x, cat.y,
      toy.x, toy.y,
      dummy.x, dummy.y,
      1000 //体力は仮の値
    ];
    this.seq_obs.push(input);
    this.seq_obs.shift();
    const input_sequence = new Float32Array(this.seq_obs.flat())
    const tensor = new ort.Tensor('float32', input_sequence, [1, this.seq_obs.length, 7]);
    const results = await session.run({"obs": tensor}); // [1, action_size, num_atoms]

    const action = results.action.data; // [action_size]
    let info = results.info ? results.info.data : null;
    return {action, info};
  }
}


class Toy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'toy');
    this.setScale(scale);
    this.cursors = scene.input.keyboard.createCursorKeys();  // 矢印キー入力
    this.currentSpeed = 1;  // 初期値は 1
  }

  setSpeed(speed) {
    this.currentSpeed = speed;
  }

  update() {
    if (this.cursors.left.isDown) {
      this.move('left');
    }
    if (this.cursors.right.isDown) {
      this.move('right');
    }
    if (this.cursors.up.isDown) {
      this.move('up');
    }
    if (this.cursors.down.isDown) {
      this.move('down');
    }

    // ボタン操作
    const direction = this.scene.activeDirection;
    if (direction) {
      this.move(direction);
    }
  }

  move(direction) {
    const matchingActions = actions_toy.filter(action => 
      action.name === direction && action.speed === this.currentSpeed
    );

    if (matchingActions.length > 0) {
      const action = matchingActions[0];
      this.x += action.dx * action.speed;
      this.y += action.dy * action.speed;
    }

    // 境界チェック
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }
}


class Dummy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'cat'); // ダミーもcat画像を流用
    this.setScale(scale * 0.8);
    this.visible = debugMode;
    this.setAlpha(debugMode ? 0.7 : 0); // デバッグ時のみ半透明で表示
    this.currentAction = 0;
  }

  move() {
    // configのactionを使ってランダム移動
    if (!actions || actions.length === 0) return;
    const actionIdx = Phaser.Math.Between(0, actions.length - 1);
    this.currentAction = actionIdx;
    const action = actions[actionIdx];
    if (action) {
      this.x += action.dx * (action.speed || 1);
      this.y += action.dy * (action.speed || 1);
    }
    // 境界チェック
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }

  setDebugVisible(flag) {
    this.visible = flag;
    this.setAlpha(flag ? 0.7 : 0);
  }
}

function generateDummyPosition(){
  //return [getRandomInt(0, environment.width), getRandomInt(0, environment.height)];
  return [getRandomInt(650, 750), getRandomInt(50, 150)];// この周辺の位置だと、パフォーマンスが良い

  function getRandomInt(min, max) {
    return Math.floor(min + Math.random() * (max - min));
  }
}

class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: 'GameScene' });
    this.catImageSize = { width: 0, height: 0}; // 初期値
    this.toyImageSize = { width: 0, height: 0}; // 初期値
    this.isImageLoaded = false; // 追加
    this.dummy = null;
    this.isHardMode = false; // デフォルトはイージーモード
  }

  preload() {
    let base = window.base_path || '/';
    if (!base.endsWith('/')) base += '/';
    this.load.image('cat', base+'cat.png');
    this.load.image('toy', base+'toy.png');
    this.load.on('filecomplete-image-cat', this.setImageSize, this);
    this.load.on('filecomplete-image-toy', this.setImageSize, this);
  }
  setImageSize(key, type, data){
    if(key === 'cat'){
      this.catImageSize.width = data.width;
      this.catImageSize.height = data.height;
    } else if(key === 'toy'){
      this.toyImageSize.width = data.width;
      this.toyImageSize.height = data.height;
    }
    this.isImageLoaded = true;
  }

  create() {
    this.add.text(400, 60, 'ねこと戯れよう！', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    }).setOrigin(0.5);

    if(!this.isImageLoaded){
      return;
    }
    //スケールを調整
    const catScale = this.calculateScale(this.catImageSize.width, this.catImageSize.height)*0.2;
    const toyScale = this.calculateScale(this.toyImageSize.width, this.toyImageSize.height);
    const init = [
      400, 400,
      100, 100,
      ...generateDummyPosition(),
      1000 // 体力の初期値（仮）
    ];
    console.log(init);
    this.cat = new Cat(this, init[0], init[1], init, catScale);
    this.toy = new Toy(this, init[2], init[3], toyScale);
    this.toy.setSpeed(1); // 初期速度を 1 に設定

    // Dummyの生成
    this.dummy = new Dummy(this, init[4], init[5], catScale);
    this.add.existing(this.cat);
    this.add.existing(this.toy);
    this.add.existing(this.dummy);

    this.dummy.setDebugVisible(debugMode);

    this.gameOver = false;
    this.gameOverText = this.add.text(400, 300, '遊んでくれた！良かったね！', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    });
    this.gameOverText.setOrigin(0.5);
    this.gameOverText.setVisible(false); // ゲームオーバーのテキストを非表示にする

    // リスタートボタン（Reactで管理するため削除）
    // this.restartButton = this.add.text(400, 380, 'リスタート', {
    //   fontSize: '32px',
    //   fill: '#fff',
    //   backgroundColor: '#00bbff',
    //   padding: { left: 20, right: 20, top: 10, bottom: 10 },
    //   borderRadius: 8,
    //   align: 'center',
    // })
    //   .setOrigin(0.5)
    //   .setInteractive()
    //   .on('pointerdown', () => {
    //     this.scene.restart();
    //   });
    // this.restartButton.setVisible(false);

    // コントロール・モード切替ボタンもReactで管理するため削除
    // this.createControlButtons();
    // this.createModeToggleButtons();

    // デバッグモード切り替えキー（例：Dキー）
    this.input.keyboard.on('keydown-D', () => {
      debugMode = !debugMode;
      this.dummy.setDebugVisible(debugMode);
    });
  }

  update() {
    if (!this.gameOver) {
      this.cat.move(this.toy, this.dummy);
      this.toy.update();
      if (this.isHardMode) {
        this.dummy.move(); // ハードモード時のみdummyが動く
      }
    }
    // 衝突判定（矩形の重なりをチェック）
    const catBounds = this.cat.getBounds();
    const toyBounds = this.toy.getBounds();

    if (!this.gameOver && Phaser.Geom.Intersects.RectangleToRectangle(catBounds, toyBounds)) {
      this.gameOver = true;
      this.gameOverText.setVisible(true);
      // Reactで管理するためrestartButtonの表示は削除
      // this.restartButton.setVisible(true);
      if (window.onGameOver) window.onGameOver();
    }
  }

  calculateScale(imageWidth, imageHeight){
    const gameWidth = this.game.config.width;
    const gameHeight = this.game.config.height;

    // 画像の幅と高さを取得
    const scaleX = gameWidth / imageWidth;
    const scaleY = gameHeight / imageHeight;

    // 画像のアスペクト比を維持しつつ、ゲーム画面に収まるようにスケールを計算
    return Math.min(scaleX, scaleY, 0.25);
  }
}


// ゲームを初期化する関数
async function initializeGame() {
  await loadConfig(); // base_pathはReact側でwindow.base_pathにセットしてから呼ぶこと
  await loadModel(); // モデルを読み込む

  // ゲーム設定
  const config = {
    type: Phaser.AUTO,
    width: environment.width,
    height: environment.height,
    parent: 'game-container',
    scene: GameScene,
  };

  // すでにPhaserインスタンスが存在する場合は破棄
  if (window.currentGame && window.currentGame.destroy) {
    window.currentGame.destroy(true);
  }
  // ゲームインスタンスの作成
  window.currentGame = new Phaser.Game(config);
}

// グローバルから呼び出せるようにする
window.initializeGame = initializeGame;