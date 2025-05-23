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

function softmax(arr) {
  const maxVal = Math.max(...arr);  // オーバーフロー対策
  const expArr = arr.map(v => Math.exp(v - maxVal)); // exp(v - maxVal)でスケーリング
  const sumExp = expArr.reduce((sum, val) => sum + val, 0);
  return expArr.map(v => v / sumExp);
}

// 猫クラス
class Cat extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, init_input, scale) {
    super(scene, x, y, 'cat');
    this.setScale(scale);
    this.seq_obs = []
    for(let seq_i=0; seq_i < model_config.sequence_length; seq_i++){
      this.seq_obs[seq_i] = init_input;
    }
    this.interest = [];
    this.interestText = scene.add.text(this.x, this.y - 20, '興味なし', {
      fontSize: '16px',
      fill: '#fff',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    });
    this.dummyPosition = [init_input[4], init_input[5]];
  }

  async move(toy) {
    const action = await this.predictAction(this, toy);
    const selectedAction = actions[action[0]][action[1]];
    if (selectedAction) {
      this.x += selectedAction.dx * selectedAction.speed;
      this.y += selectedAction.dy * selectedAction.speed;
    }
  
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);

    // interest に基づいたテキストの更新
    this.interestText.setPosition(this.x + 20, this.y - 40);
    const interestTextMap = {
      0: '興味なし',
      1: '探索中',
      2: '興味津々'
    };
    const index = this.interest.indexOf(Math.max(...this.interest))
    this.interestText.setText(interestTextMap[index] + (this.interest[index]).toFixed(3));
  }
  async predictAction(cat, toy) {
    if (!session) throw new Error('Model not loaded yet!');

    const input = [
      cat.x, cat.y,
      toy.x, toy.y,
      ...this.dummyPosition
    ];
    this.seq_obs.unshift(input)
    this.seq_obs.pop()
    const input_sequence = new Float32Array(this.seq_obs.flat())
    const tensor = new ort.Tensor('float32', input_sequence, [1, this.seq_obs.length, 6]);
    const results = await session.run({"obs": tensor}); // [1, action_size, num_atoms]
    // interest の取得と更新（動きの大きさで興味を計測する）
    this.interest = results.q_values_speed.data; // [action_size]
    // 最大のQ値を持つ行動
    return [results.action_speed.data, results.action_direction.data];
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

function generateDummyPosition(){
  return [getRandomInt(environment.width), getRandomInt(environment.height)];

  function getRandomInt(max) {
    return Math.floor(Math.random() * max);
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
      ...generateDummyPosition()
    ];
    console.log(init);
    this.cat = new Cat(this, init[0], init[1], init, catScale);
    this.toy = new Toy(this, init[2], init[3], toyScale);
    this.toy.setSpeed(1); // 初期速度を 1 に設定

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
    this.createControlButtons() 
    }

  update() {
    if (!this.gameOver) {
      this.cat.move(this.toy);
      this.toy.update();  // Toyの移動処理を呼び出す
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

  createControlButtons() {
    const buttonSize = 50;
    const buttonOffset = 10;
    const baseX = 650;
    const baseY = 450;

    const directions = ['up', 'down', 'left', 'right'];
    const buttonPositions = {
      up: { x: baseX, y: baseY - buttonSize - buttonOffset },
      down: { x: baseX, y: baseY + buttonSize + buttonOffset },
      left: { x: baseX - buttonSize - buttonOffset, y: baseY },
      right: { x: baseX + buttonSize + buttonOffset, y: baseY }
    };

    this.activeDirection = null;

    directions.forEach((direction) => {
      const { x, y } = buttonPositions[direction];

      const button = this.add.rectangle(x, y, buttonSize, buttonSize, 0x00ff00)
        .setInteractive()
        .on('pointerdown', () => {
          this.activeDirection = direction;
        })
        .on('pointerup', () => {
          this.activeDirection = null;
        })
        .on('pointerout', () => {
          this.activeDirection = null;
        });

      const label = {
        up: '↑',
        down: '↓',
        left: '←',
        right: '→'
      }[direction];

      this.add.text(x, y, label, { fontSize: '24px', color: '#000' }).setOrigin(0.5);
    });

    // 速度切り替えボタン
    const speedButton1 = this.add.rectangle(600, 550, buttonSize, buttonSize, 0x0000ff)
      .setInteractive()
      .on('pointerdown', () => {
        this.toy.setSpeed(1);
      });

    this.add.text(600, 550, '1', { fontSize: '24px', color: '#fff' }).setOrigin(0.5);

    const speedButton2 = this.add.rectangle(700, 550, buttonSize, buttonSize, 0xff0000)
      .setInteractive()
      .on('pointerdown', () => {
        this.toy.setSpeed(2.5);
      });

    this.add.text(700, 550, '2.5', { fontSize: '24px', color: '#fff' }).setOrigin(0.5);
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