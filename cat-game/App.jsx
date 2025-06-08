import React, { useState } from 'react';
import Home from './Home.jsx';
import Game from './Game.jsx';

export default function App() {
  const [page, setPage] = useState('home');
  const handleStart = () => setPage('game');
  const handleRestart = () => setPage('game');
  const handleHome = () => setPage('home');

  return (
    <div style={{ minHeight: '100vh', background: '#fff' }}>
      {page === 'home' && <Home onStart={handleStart} />}
      {page === 'game' && <Game onRestart={handleHome} />}
    </div>
  );
}
