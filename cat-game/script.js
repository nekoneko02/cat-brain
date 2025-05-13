let session;

async function loadModel() {
  try {
    console.log('Loading model...');
    session = await ort.InferenceSession.create('./cat_dqn_policy.onnx');
    console.log('Model loaded:', session);
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}


let actions = [];
let observation_space = {};
let environment = {};
let hidden_state = null;

async function loadConfig() {
  const response = await fetch('config/common.json'); // JSONファイルのパス
  const data = await response.json();
  actions = data.actions.cat;
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

function sum(probabilities, z_support) {
  // probabilities は [num_actions, num_atoms] の形
  const num_actions = actions.length;
  const result = new Array(num_actions);
  for (let action_i = 0; action_i < num_actions; action_i++) {
    let sum = 0;
    for (let atom_i = 0; atom_i < z_support.length; atom_i++) {
        sum += probabilities[action_i * z_support.length + atom_i] * z_support[atom_i];
    }
    result[action_i] = sum;
  }
  return result;
}

// 猫クラス
class Cat extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'cat');
    this.setScale(scale);
    this.z_support = linspace(model_config.v_min, model_config.v_max, model_config.num_atoms);
  }

  async move(toy) {
    const action = await this.predictAction(this, toy);
    const selectedAction = actions[action];
    if (selectedAction) {
      this.x += selectedAction.dx;
      this.y += selectedAction.dy;
    }
  
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }
  async predictAction(cat, toy) {
    if (!session) throw new Error('Model not loaded yet!');

    const input = new Float32Array([
      cat.x, cat.y,
      toy.x, toy.y,
      400,300
    ]);

    const tensor = new ort.Tensor('float32', input, [1, 1, 6]);
    const results = await session.run({"obs": tensor}); // [1, action_size, num_atoms]
    const output = sum(results.probabilities.data, this.z_support); // [action_size]
    // 最大のQ値を持つ行動
    const maxIdx = output.indexOf(Math.max(...output));
    return maxIdx;
  }
}


// おもちゃクラス
class Toy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'toy');
    this.isDragging = false;
    this.offsetX = 0; // ドラッグ開始時のねこじゃらしとマウスのx座標の差を保持
    this.offsetY = 0; 

    this.setInteractive();
    // 画像の縮尺
    this.setScale(scale);

    this.on('pointerdown', (pointer) => {
      this.isDragging = true;
      this.offsetX = pointer.x - this.x; // マウスとねこじゃらしの座標の差を計算
      this.offsetY = pointer.y - this.y; 
    });

    this.on('pointermove', (pointer) => {
      if (this.isDragging) {
        this.x = pointer.x - this.offsetX; // 差を考慮してねこじゃらしの位置を更新
        this.y = pointer.y - this.offsetY;
        // 境界チェック
        this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
        this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
      }
    });

    this.on('pointerup', () => {
      this.isDragging = false;
    });
  }
}

// ゲームシーン
class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: 'GameScene' });
    this.catImageSize = { width: 0, height: 0}; // 初期値
    this.toyImageSize = { width: 0, height: 0}; // 初期値
    this.isImageLoaded = false; // 追加
  }

  preload() {
    this.load.image('cat', 'cat.png');
    this.load.image('toy', 'toy.png');
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
    this.add.text(400, 60, 'ねこから逃げろ', {
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
    this.cat = new Cat(this, 400, 400, catScale);
    this.toy = new Toy(this, 100, 100, toyScale);
    
    this.add.existing(this.cat);
    this.add.existing(this.toy);

    this.gameOver = false;
    this.gameOverText = this.add.text(400, 300, 'ゲームオーバー', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    });
    this.gameOverText.setOrigin(0.5);
    this.gameOverText.setVisible(false); // ゲームオーバーのテキストを非表示にする
  }

  update() {
    if (this.cat && this.toy && !this.gameOver) {
      this.cat.move(this.toy);
    }
    // 衝突判定（矩形の重なりをチェック）
    const catBounds = this.cat.getBounds();
    const toyBounds = this.toy.getBounds();

    if (Phaser.Geom.Intersects.RectangleToRectangle(catBounds, toyBounds)) {
      this.gameOver = true;
      this.gameOverText.setVisible(true);
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
  await loadConfig(); // 設定を読み込むまで待機
  await loadModel(); // モデルを読み込む

  // ゲーム設定
  const config = {
    type: Phaser.AUTO,
    width: environment.width,
    height: environment.height,
    parent: 'game-container',
    scene: GameScene,
  };

  // ゲームインスタンスの作成
  const game = new Phaser.Game(config);
}

// ゲームを初期化
initializeGame();