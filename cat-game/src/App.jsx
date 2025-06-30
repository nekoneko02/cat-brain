import React, { useState } from 'react';
import StartScreen from './StartScreen';
import GameScreen from './GameScreen';
import './App.css';

function App() {
  const [started, setStarted] = useState(false);
  window.base_path = import.meta.env.BASE_URL;
  return (
    <div>
      {!started ? (
        <StartScreen onStart={() => setStarted(true)} />
      ) : (
        <GameScreen />
      )}
    </div>
  );
}

export default App;
