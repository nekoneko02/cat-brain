import React, { useState, useEffect } from 'react';

export default function StartScreen({ onStart }) {
  const [modelId, setModelId] = useState('');
  const [modelList, setModelList] = useState([]);

  useEffect(() => {
    fetch('/models/models_index.json')
      .then(res => res.json())
      .then(list => {
        setModelList(list);
        setModelId(list[0]?.id || '');
      });
  }, []);

  const handleStart = () => {
    window.catModelId = modelId;
    onStart();
  };

  const selectedModel = modelList.find(m => m.id === modelId);

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
          <li>速度はスライダー、Shiftキー（長押しで高速）、またはアナログスティックの倒し具合で調整できます</li>
        </ul>
      </section>
      <div style={{ margin: '30px 0' }}>
        <label style={{ fontSize: 18, marginRight: 12 }}>モデル選択:</label>
        <select value={modelId} onChange={e => setModelId(e.target.value)} style={{ fontSize: 18 }}>
          {modelList.map(model => (
            <option key={model.id} value={model.id}>{model.name}</option>
          ))}
        </select>
        {selectedModel && (
          <div style={{ marginTop: 12, fontSize: 16, color: '#555' }}>
            <b>ねこの性格:</b> {selectedModel.description}
          </div>
        )}
      </div>
      <button style={{ fontSize: 24, padding: '12px 40px', borderRadius: 8 }} onClick={handleStart}>
        スタート
      </button>
    </div>
  );
}
