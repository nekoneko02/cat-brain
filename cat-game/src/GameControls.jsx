import React, { useRef, useState } from 'react';

export default function GameControls({ onDirection, onSpeed, onMode, isHardMode, speed }) {
  const stickRef = useRef(null);
  const dragging = useRef(false);
  const [stickPos, setStickPos] = useState({ x: 60, y: 60 });

  // アナログスティックの中心座標
  const center = { x: 60, y: 60 };
  const radius = 50;
  const knobRadius = 18;

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
    // 4方向判定
    let dir = null;
    if (dist > 20) {
      if (Math.abs(dx) > Math.abs(dy)) {
        dir = dx > 0 ? 'right' : 'left';
        x = center.x + Math.sign(dx) * Math.min(Math.abs(dx), radius - knobRadius);
        y = center.y;
      } else {
        dir = dy > 0 ? 'down' : 'up';
        x = center.x;
        y = center.y + Math.sign(dy) * Math.min(Math.abs(dy), radius - knobRadius);
      }
      setStickPos({ x, y });
      onDirection(dir);
    } else {
      setStickPos(center);
      onDirection(null);
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
          width={120}
          height={120}
          style={{
            touchAction: 'none',
            background: '#f5f5f5',
            borderRadius: '50%',
            boxShadow: '0 0 8px #aaa',
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
      {/* 速度・モード切替ボタン */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ marginBottom: 16 }}>
          <button
            onClick={() => onSpeed(1)}
            style={{ ...wideBtnStyle, background: speed === 1 ? '#0000ff' : '#ccc', color: '#fff' }}
          >
            速度1
          </button>
          <button
            onClick={() => onSpeed(2.5)}
            style={{
              ...wideBtnStyle,
              background: speed === 2.5 ? '#ff0000' : '#ccc',
              color: '#fff',
              marginLeft: 8,
            }}
          >
            速度2.5
          </button>
        </div>
        <div>
          <button
            onClick={() => onMode(true)}
            style={{ ...wideBtnStyle, background: isHardMode ? '#ff8800' : '#888', fontSize: 16 }}
          >
            ハードモード
          </button>
          <button
            onClick={() => onMode(false)}
            style={{
              ...wideBtnStyle,
              background: !isHardMode ? '#00bbff' : '#888',
              marginLeft: 8,
              fontSize: 16,
            }}
          >
            イージーモード
          </button>
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
