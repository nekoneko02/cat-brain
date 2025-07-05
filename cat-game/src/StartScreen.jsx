import React, { useState } from 'react';

export default function StartScreen({ onStart }) {
  return (
    <div style={{ textAlign: 'center', marginTop: 60 }}>
      <h1>ねこゲーム</h1>
      <h2>～AIねこと遊ぼう！～</h2>
      <section style={{ margin: '30px 0' }}>
        <h3>ゲーム概要</h3>
        <p>
          このゲームは、AIで動く「ねこ」とプレイヤーが操作する「おもちゃ」が登場します。
          <br />
          ねこはAIモデルによって自律的に動きます。
          <br />
          プレイヤーは矢印キーまたは画面のボタンでおもちゃを操作し、ねこと遊びましょう！
        </p>
      </section>
      <section style={{ margin: '30px 0' }}>
        <h3>操作説明</h3>
        <ul style={{ display: 'inline-block', textAlign: 'left' }}>
          <li>矢印キーまたは画面のボタンでおもちゃを動かせます</li>
          <li>速度切替ボタンでおもちゃの速さを変更できます</li>
          <li>モード切替で難易度を変更できます</li>
        </ul>
      </section>
      <button style={{ fontSize: 24, padding: '12px 40px', borderRadius: 8 }} onClick={onStart}>
        スタート
      </button>
    </div>
  );
}
