let session;

async function loadModel() {
  session = await ort.InferenceSession.create('cat_dqn_policy.onnx');
  console.log(session)
}
loadModel();

async function predictAction(cat, toy) {
  if (!session) throw new Error('Model not loaded yet!');

  const input = new Float32Array([
    toy.x, toy.y,
    cat.x, cat.y,
  ]);

  const tensor = new ort.Tensor('float32', input, [1, 4]);

  const results = await session.run({ obs: tensor });
  const output = results.q_values.data; // Q値

  // 最大のQ値を持つ行動
  const maxIdx = output.indexOf(Math.max(...output));
  return maxIdx;
}


// 猫クラス
class Cat extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'cat');
    this.speed = 2;
    this.setScale(scale);
  }

  async move(toy) {
    const action = await predictAction(this, toy);

    if (action === 0) {
      this.y -= this.speed; // �?
    } else if (action === 1) {
      this.y += this.speed; // �?
    } else if (action === 2) {
      this.x -= this.speed; // 左
    } else if (action === 3) {
      this.x += this.speed; // 右
    }

    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
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

// ゲーム設定
const config = {
  type: Phaser.AUTO,
  width: 800,
  height: 600,
  parent: 'game-container',
  scene: GameScene,
};

// ゲームインスタンスの作成
const game = new Phaser.Game(config);
