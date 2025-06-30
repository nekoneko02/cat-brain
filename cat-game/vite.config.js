import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const version = process.env.VERSION;
const base = version ? `/cat-game/${version}` : '/';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base
})
