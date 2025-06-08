# ねこと遊ぶゲーム（cat-game）

## 開発用（ローカル開発サーバー）

1. 必要なパッケージをインストール

```
npm install
```

2. 開発サーバーを起動

```
npx run dev
```

3. ブラウザで `http://localhost:5173` などにアクセス


## リリース用（本番ビルド）

1. BASE_URL 環境変数を指定して本番ビルドを作成（例: S3の /v7/ 配下に配置する場合）

```
BASE_URL=v7/ npx vite build cat-game
```

2. `dist/` ディレクトリが生成されるので、これをWebサーバーやS3バケットの `/v7/` 配下にデプロイ

3. dist配下の静的ファイルと追加リソースをまとめてコピー

```
cp -r ./cat-game/dist/* /mnt/c/Users/takaf/Downloads/cp-cat-game/v{n}/
cp ./cat-game/cat_dqn_policy.onnx ./cat-game/cat.png ./cat-game/toy.png /mnt/c/Users/takaf/Downloads/cp-cat-game/v{n}/
```


## 注意
- React/JSXを使う場合はVite等のビルドツールが必須です。
- `index.jsx` ではなく `index.js` などにビルドされます。
- 静的ファイルとして配信する場合は `dist/` 配下のみをWebサーバーで公開してください。

---

### 依存
- Node.js, npm
- Vite（`npx vite` で自動インストール可）
- Phaser, React など（`npm install` で自動インストール）

---

### 参考
- [Vite公式ドキュメント](https://vitejs.dev/)
- [React公式ドキュメント](https://ja.react.dev/)
