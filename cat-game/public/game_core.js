// game_core.js (ESM)

// グローバル変数の多重定義を防ぐ
globalThis.catGameGlobals = globalThis.catGameGlobals || {};
export const catGameGlobals = globalThis.catGameGlobals;

export function linspace(v_min, v_max, num_atoms) {
  const arr = new Array(num_atoms);
  for (let i = 0; i < num_atoms; i++) {
      arr[i] = v_min + (v_max - v_min) * (i / (num_atoms - 1));
  }
  return arr;
}

export function softmax(arr, temperature = 1.0) {
  const maxVal = Math.max(...arr);
  const expArr = arr.map(v => Math.exp((v - maxVal) / temperature));
  const sumExp = expArr.reduce((sum, val) => sum + val, 0);
  return expArr.map(v => v / sumExp);
}

export function generateDummyPosition() {
  return [getRandomInt(catGameGlobals.environment.width), getRandomInt(catGameGlobals.environment.height)];
  function getRandomInt(max) {
    return Math.floor(Math.random() * max);
  }
}

export class Cat extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, init_input, scale) {
    super(scene, x, y, 'cat');
    this.setScale(scale);
    this.seq_obs = [];
    for(let seq_i=0; seq_i < catGameGlobals.model_config.sequence_length; seq_i++){
      this.seq_obs[seq_i] = init_input;
    }
    this.interest = [];
    this.dummyPosition = [init_input[4], init_input[5]];
    this.infoCircle = scene.add.circle(0, 0, 10, 0xffff00, 0.7);
    this.infoCircle.setVisible(false);
  }
  async move(toy, dummy) {
    const {action, info} = await this.predictAction(this, toy, dummy);
    const selectedAction = catGameGlobals.actions[action];
    if (selectedAction) {
      this.x += selectedAction.dx * selectedAction.speed;
      this.y += selectedAction.dy * selectedAction.speed;
    }
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
    if (catGameGlobals.debugMode && info && info.length >= 2) {
      let seq = info.length / 2;
      if (Number.isInteger(seq) && seq > 1) {
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
        if (!this.infoCircle) {
          this.infoCircle = this.scene.add.circle(0, 0, 10, 0xffff00, 0.7);
        }
        this.infoCircle.setPosition(info[0], info[1]);
        this.infoCircle.setVisible(true);
      }
    } else {
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
    if (!catGameGlobals.session) throw new Error('Model not loaded yet!');
    const input = [cat.x, cat.y, toy.x, toy.y, dummy.x, dummy.y];
    this.seq_obs.push(input);
    this.seq_obs.shift();
    const input_sequence = new Float32Array(this.seq_obs.flat());
    const tensor = new ort.Tensor('float32', input_sequence, [1, this.seq_obs.length, 6]);
    const results = await catGameGlobals.session.run({"obs": tensor});
    this.interest = results.q_values.data;
    let info = results.info ? results.info.data : null;
    let temperature = 0.1;
    let probs = softmax(this.interest, temperature);
    let action = 0;
    let r = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r < acc) {
        action = i;
        break;
      }
    }
    return {action, info};
  }
}

export class Toy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'toy');
    this.setScale(scale);
    this.cursors = scene.input.keyboard.createCursorKeys(); // ここだけ
    this.currentSpeed = 1;
  }
  setSpeed(speed) {
    this.currentSpeed = speed;
  }
  update() {
    if (this.cursors.left.isDown) this.move('left');
    if (this.cursors.right.isDown) this.move('right');
    if (this.cursors.up.isDown) this.move('up');
    if (this.cursors.down.isDown) this.move('down');
    const direction = this.scene.activeDirection;
    if (direction) this.move(direction);
  }
  move(direction) {
    const matchingActions = catGameGlobals.actions_toy.filter(action => action.name === direction && action.speed === this.currentSpeed);
    if (matchingActions.length > 0) {
      const action = matchingActions[0];
      this.x += action.dx * action.speed;
      this.y += action.dy * action.speed;
    }
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }
}

export class Dummy extends Phaser.GameObjects.Sprite {
  constructor(scene, x, y, scale) {
    super(scene, x, y, 'cat');
    this.setScale(scale * 0.8);
    this.visible = catGameGlobals.debugMode;
    this.setAlpha(catGameGlobals.debugMode ? 0.7 : 0);
    this.currentAction = 0;
  }
  move() {
    if (!catGameGlobals.actions || catGameGlobals.actions.length === 0) return;
    const actionIdx = Phaser.Math.Between(0, catGameGlobals.actions.length - 1);
    this.currentAction = actionIdx;
    const action = catGameGlobals.actions[actionIdx];
    if (action) {
      this.x += action.dx * (action.speed || 1);
      this.y += action.dy * (action.speed || 1);
    }
    this.x = Phaser.Math.Clamp(this.x, 0, this.scene.game.config.width - this.displayWidth);
    this.y = Phaser.Math.Clamp(this.y, 0, this.scene.game.config.height - this.displayHeight);
  }
  setDebugVisible(flag) {
    this.visible = flag;
    this.setAlpha(flag ? 0.7 : 0);
  }
}

// GameSceneクラスをgame_core.jsに移動
export class GameScene extends Phaser.Scene {
  constructor() {
    super({ key: 'GameScene' });
    this.catImageSize = { width: 0, height: 0}; // 初期値
    this.toyImageSize = { width: 0, height: 0}; // 初期値
    this.isImageLoaded = false; // 追加
    this.dummy = null;
    this.isHardMode = false; // デフォルトはイージーモード
  }

  preload() {
    console.log('[GameScene] preload start');
    this.load.image('cat', 'cat.png');
    this.load.image('toy', 'toy.png');
    this.load.on('filecomplete-image-cat', this.setImageSize, this);
    this.load.on('filecomplete-image-toy', this.setImageSize, this);
    this.load.on('complete', () => {
      console.log('[GameScene] preload complete');
    });
  }
  setImageSize(key, type, texture){
    console.log('[GameScene] setImageSize', key, texture);
    if(key === 'cat'){
      this.catImageSize.width = texture.source[0].width;
      this.catImageSize.height = texture.source[0].height;
    } else if(key === 'toy'){
      this.toyImageSize.width = texture.source[0].width;
      this.toyImageSize.height = texture.source[0].height;
    }
    if (this.catImageSize.width && this.toyImageSize.width) {
      this.isImageLoaded = true;
      console.log('[GameScene] isImageLoaded true');
    }
  }

  create() {
    console.log('[GameScene] create called');
    this.add.text(400, 60, 'ねこと戯れよう！', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    }).setOrigin(0.5);

    const catScale = this.calculateScale(this.catImageSize.width, this.catImageSize.height)*0.2;
    const toyScale = this.calculateScale(this.toyImageSize.width, this.toyImageSize.height);
    const init = [
      400, 400,
      100, 100,
      ...generateDummyPosition()
    ];
    console.log('[GameScene] create init', init);
    this.cat = new Cat(this, init[0], init[1], init, catScale);
    this.toy = new Toy(this, init[2], init[3], toyScale);
    this.toy.setSpeed(1);
    console.log('[GameScene] Cat instance', this.cat);
    this.dummy = new Dummy(this, init[4], init[5], catScale);
    this.add.existing(this.cat);
    this.add.existing(this.toy);
    this.add.existing(this.dummy);
    this.dummy.setDebugVisible(window.catGameGlobals.debugMode);
    this.gameOver = false;
    this.gameOverText = this.add.text(400, 300, '遊んでくれた！良かったね！', {
      fontSize: '48px',
      fill: '#f00',
      fontFamily: '"Noto Sans JP", "Meiryo", sans-serif'
    });
    this.gameOverText.setOrigin(0.5);
    this.gameOverText.setVisible(false);
    this.createControlButtons();
    this.createModeToggleButtons();
    this.input.keyboard.on('keydown-D', () => {
      window.catGameGlobals.debugMode = !window.catGameGlobals.debugMode;
      this.dummy.setDebugVisible(window.catGameGlobals.debugMode);
    });
    console.log('[GameScene] create end');
  }

  update() {
    console.log('[GameScene] update called');
    if (!this.gameOver) {
      this.cat.move(this.toy, this.dummy);
      this.toy.update();
      if (this.isHardMode) {
        this.dummy.move();
      }
    }
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
    const scaleX = gameWidth / imageWidth;
    const scaleY = gameHeight / imageHeight;
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

  createModeToggleButtons() {
    const buttonWidth = 120;
    const buttonHeight = 40;
    const baseX = 120;
    const baseY = 550;
    this.hardModeButton = this.add.rectangle(baseX, baseY, buttonWidth, buttonHeight, this.isHardMode ? 0xff8800 : 0x888888)
      .setInteractive()
      .on('pointerdown', () => {
        this.isHardMode = true;
        this.updateModeButtonStyles();
      });
    this.hardModeLabel = this.add.text(baseX, baseY, 'ハードモード', { fontSize: '20px', color: '#fff' }).setOrigin(0.5);
    this.easyModeButton = this.add.rectangle(baseX + buttonWidth + 20, baseY, buttonWidth, buttonHeight, !this.isHardMode ? 0x00bbff : 0x888888)
      .setInteractive()
      .on('pointerdown', () => {
        this.isHardMode = false;
        this.updateModeButtonStyles();
      });
    this.easyModeLabel = this.add.text(baseX + buttonWidth + 20, baseY, 'イージーモード', { fontSize: '20px', color: '#fff' }).setOrigin(0.5);
  }

  updateModeButtonStyles() {
    if (this.hardModeButton && this.easyModeButton) {
      this.hardModeButton.setFillStyle(this.isHardMode ? 0xff8800 : 0x888888);
      this.easyModeButton.setFillStyle(!this.isHardMode ? 0x00bbff : 0x888888);
    }
  }
}
