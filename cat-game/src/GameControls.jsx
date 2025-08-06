import React, { useRef, useState } from 'react';

export default function GameControls({ onDirection, onSpeed, onMode, isHardMode, speed }) {
  // Shift長押しで速度1、通常時は0.5
  // 初回レンダリング時のみ速度0.5に設定
  React.useEffect(() => {
    onSpeed(0.5);
    // eslint-disable-next-line
  }, []);

  React.useEffect(() => {
    let shiftDown = false;
    const handleKeyDown = (e) => {
      if (e.key === 'Shift') {
        if (!shiftDown) {
          shiftDown = true;
          onSpeed(1);
        }
      }
    };
    const handleKeyUp = (e) => {
      if (e.key === 'Shift') {
        shiftDown = false;
        onSpeed(0.5);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [onSpeed]);
  const stickRef = useRef(null);
  const dragging = useRef(false);
  // スティックサイズを画面幅に応じて可変（最大180px）
  const getStickSize = () => {
    if (typeof window !== 'undefined') {
      return Math.min(Math.floor(window.innerWidth * 0.25), 180);
    }
    return 180;
  };
  const [stickSize, setStickSize] = useState(getStickSize());
  React.useEffect(() => {
    const handleResize = () => setStickSize(getStickSize());
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  const center = { x: stickSize / 2, y: stickSize / 2 };
  const radius = stickSize / 2 - 10;
  const knobRadius = Math.max(18, stickSize * 0.15);
  const [stickPos, setStickPos] = useState(center);

  // スティック操作イベント
  const handlePointerDown = (e) => {
    dragging.current = true;
    handlePointerMove(e);
  };
  const handlePointerUp = () => {
    dragging.current = false;
    setStickPos(center);
    onDirection(null);
  };
  const handlePointerMove = (e) => {
    if (!dragging.current) return;
    const rect = stickRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    let x = clientX - rect.left;
    let y = clientY - rect.top;
    let dx = x - center.x;
    let dy = y - center.y;
    let dist = Math.sqrt(dx * dx + dy * dy);
    const maxDist = radius - knobRadius;
    // 連続値ベクトル
    let vx = 0, vy = 0;
    if (dist > 5) {
      // 範囲内に収める
      if (dist > maxDist) {
        dx = (dx / dist) * maxDist;
        dy = (dy / dist) * maxDist;
        x = center.x + dx;
        y = center.y + dy;
        dist = maxDist;
      }
      setStickPos({ x, y });
      vx = dx / maxDist;
      vy = dy / maxDist;
      // -1.0～1.0に正規化
      vx = Math.max(-1, Math.min(1, vx));
      vy = Math.max(-1, Math.min(1, vy));
      onDirection({ x: vx, y: vy });
      // スティックの倒し具合で速度を切り替え（0～2で線形割り当て）
      const stickSpeed = Math.min(dist / maxDist, 1) * 2;
      onSpeed(Number(stickSpeed.toFixed(2)));
    } else {
      setStickPos(center);
      onDirection(null);
      onSpeed(0);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'flex-start',
        marginTop: 0,
      }}
    >
      {/* アナログスティック */}
      <div style={{ marginRight: 32, userSelect: 'none' }}>
        <svg
          ref={stickRef}
          width={stickSize}
          height={stickSize}
          style={{
            touchAction: 'none',
            background: '#f5f5f5',
            borderRadius: '50%',
            boxShadow: '0 0 8px #aaa',
            transition: 'width 0.2s, height 0.2s',
          }}
          onMouseDown={handlePointerDown}
          onMouseUp={handlePointerUp}
          onMouseLeave={handlePointerUp}
          onMouseMove={handlePointerMove}
          onTouchStart={handlePointerDown}
          onTouchEnd={handlePointerUp}
          onTouchCancel={handlePointerUp}
          onTouchMove={handlePointerMove}
        >
          <circle
            cx={center.x}
            cy={center.y}
            r={radius}
            fill="#ddd"
            stroke="#aaa"
            strokeWidth={3}
          />
          <circle cx={stickPos.x} cy={stickPos.y} r={knobRadius} fill="#bbb" />
        </svg>
      </div>
      {/* 速度・モード切替スライダー */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ marginBottom: 16, width: 180 }}>
          <label htmlFor="speed-slider" style={{ fontSize: 16, marginBottom: 8, display: 'block', textAlign: 'center' }}>
            速度: {speed.toFixed(2)}
          </label>
          <input
            id="speed-slider"
            type="range"
            min={0}
            max={2}
            step={0.01}
            value={speed}
            onChange={e => onSpeed(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      </div>
    </div>
  );
}

const wideBtnStyle = {
  fontSize: 18,
  width: 110,
  height: 48,
  margin: 4,
  borderRadius: 8,
  border: 'none',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};
