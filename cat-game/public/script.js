let debugMode = false; // デバッグモードフラグ
// velocity sequence length (Pythonと合わせる)
const vel_seq_len = 10;
// let session;

async function loadModel() {
  try {
    let base = window.base_path || '/';
    if (!base.endsWith('/')) base += '/';
    // モデルIDをReact側でセット（例: window.catModelId = '1'）
    const modelId = window.catModelId || '1';
    const modelPath = `${base}models/${modelId}/policy.onnx`;
    console.log('Loading model:', modelPath);
    session = await ort.InferenceSession.create(modelPath);
    console.log('Model loaded:', session);
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}


// ゲーム環境の設定
const environment = {
  width: 800,
  height: 600
};

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
    // velocity sequence (Catは1つだけ保持)
    this.vel_x = 0.0;
    this.vel_y = 0.0;
  }

  async move(toy) {
    const { option, action } = await this.predictAction(this, toy);

    if (action) {
      // 速度を更新
      this.vel_x = action[0];
      this.vel_y = action[1];
      this.x += this.vel_x;
      this.y += this.vel_y;
    }

    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);

  }
  async predictAction(cat, toy) {
    if (!session) throw new Error('Model not loaded yet!');

    // 7次元: 相対位置2, chaser速度2, runner速度2, fatigue1
    // runner velocity sequence (flatten)
    const rel_x = Math.tanh((toy.x - cat.x) / 2000);
    const rel_y = Math.tanh((toy.y - cat.y) / 2000);
    const chaser_vel_x = cat.vel_x;
    const chaser_vel_y = cat.vel_y;
    // flatten toy.vel_seq: [vx1, vy1, vx2, vy2, ...]
    let runner_vel_seq_flat = [];
    if (toy.vel_seq && toy.vel_seq.length === vel_seq_len) {
      for (let i = 0; i < vel_seq_len; i++) {
        runner_vel_seq_flat.push(toy.vel_seq[i][0]);
        runner_vel_seq_flat.push(toy.vel_seq[i][1]);
      }
    } else {
      // fallback: 現在の速度を繰り返し
      for (let i = 0; i < vel_seq_len; i++) {
        runner_vel_seq_flat.push(toy.vel_x);
        runner_vel_seq_flat.push(toy.vel_y);
      }
    }
    const fatigue = 1.0; // 仮値
    // obs: [rel_x, rel_y, chaser_vel_x, chaser_vel_y, ...runner_vel_seq_flat..., fatigue]
    const obs = [rel_x, rel_y, chaser_vel_x, chaser_vel_y, ...runner_vel_seq_flat, fatigue];
    const obs_dim = 4 + vel_seq_len * 2 + 1;
    const tensor_obs = new ort.Tensor('float32', obs, [1, obs_dim]);

    const results = await session.run({
      "obs": tensor_obs
    });

    const option = results.option.data;
    const action = results.action.data;
    return { option, action };
  }
}


class Toy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'toy');
    this.setScale(scale);
    this.cursors = scene.input.keyboard.createCursorKeys();  // 矢印キー入力
    this.currentSpeed = 1;  // 初期値は 1
    // velocity sequence
    this.vel_seq = [];
    for (let i = 0; i < vel_seq_len; i++) {
      this.vel_seq.push([0.0, 0.0]);
    }
    this.vel_x = 0.0;
    this.vel_y = 0.0;
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
    const actions_toy = {
      'up': { dx: 0, dy: -1 },
      'down': { dx: 0, dy: 1 },
      'left': { dx: -1, dy: 0 },
      'right': { dx: 1, dy: 0 }
    };

    const action = actions_toy[direction];
    if (action) {
      // 速度を更新
      this.vel_x = action.dx * this.currentSpeed;
      this.vel_y = action.dy * this.currentSpeed;
      this.x += this.vel_x;
      this.y += this.vel_y;
      // velocity sequence更新
      this.vel_seq.push([this.vel_x, this.vel_y]);
      if (this.vel_seq.length > vel_seq_len) {
        this.vel_seq.shift();
      }
    }

    // 境界チェック
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }
}

class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: 'GameScene' });
    this.catImageSize = { width: 0, height: 0 }; // 初期値
    this.toyImageSize = { width: 0, height: 0 }; // 初期値
    this.isImageLoaded = false; // 追加
    this.isHardMode = false; // デフォルトはイージーモード
  }

  preload() {
    let base = window.base_path || '/';
    if (!base.endsWith('/')) base += '/';
    this.load.image('cat', base + 'cat.png');
    this.load.image('toy', base + 'toy.png');
    this.load.on('filecomplete-image-cat', this.setImageSize, this);
    this.load.on('filecomplete-image-toy', this.setImageSize, this);
  }
  setImageSize(key, type, data) {
    if (key === 'cat') {
      this.catImageSize.width = data.width;
      this.catImageSize.height = data.height;
    } else if (key === 'toy') {
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

    if (!this.isImageLoaded) {
      return;
    }
    //スケールを調整
    const catScale = this.calculateScale(this.catImageSize.width, this.catImageSize.height) * 0.2;
    const toyScale = this.calculateScale(this.toyImageSize.width, this.toyImageSize.height);
    // 7次元: 相対位置2, chaser速度2, runner速度2, fatigue1
    const rel_x = 100 - 400;
    const rel_y = 100 - 400;
    const chaser_vel_x = 0.0;
    const chaser_vel_y = 0.0;
    const runner_vel_x = 0.0;
    const runner_vel_y = 0.0;
    const fatigue = 1.0;
    const init = [rel_x, rel_y, chaser_vel_x, chaser_vel_y, runner_vel_x, runner_vel_y, fatigue];
    this.cat = new Cat(this, 400, 400, init, catScale);
    this.toy = new Toy(this, 100, 100, toyScale);
    this.toy.setSpeed(0.5); // 初期速度を 1 に設定

    this.add.existing(this.cat);
    this.add.existing(this.toy);

    this.gameOver = false;
    this.gameOverText = this.add.text(400, 300, '遊んでくれた！良かったね！', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    });
    this.gameOverText.setOrigin(0.5);
    this.gameOverText.setVisible(false); // ゲームオーバーのテキストを非表示にする

    // デバッグモード切り替えキー（例：Dキー）
    this.input.keyboard.on('keydown-D', () => {
      debugMode = !debugMode;
    });
  }

  update() {
    if (!this.gameOver) {
      this.cat.move(this.toy);
      this.toy.update();
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

  calculateScale(imageWidth, imageHeight) {
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