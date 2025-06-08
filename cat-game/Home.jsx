import React from 'react';

export default function Home({ onStart }) {
  return (
    <div style={{ textAlign: 'center', marginTop: 80 }}>
      <h1 style={{ fontSize: 48, color: '#f00', fontFamily: 'Noto Sans JP, Meiryo, sans-serif' }}>ねこと遊ぶゲーム</h1>
      <p style={{ fontSize: 24, margin: '40px 0' }}>
        かわいいねこと一緒に遊ぼう！<br />
        <span style={{ color: '#888', fontSize: 18 }}>（Dキーでデバッグモード切替、ハード/イージーモードも選べます）</span>
      </p>
      <button
        style={{ fontSize: 28, padding: '16px 48px', background: '#ff8800', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
        onClick={onStart}
      >
        ゲームスタート
      </button>
    </div>
  );
}
