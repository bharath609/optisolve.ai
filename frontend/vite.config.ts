import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // relative asset paths: works on Vercel root, GitHub Pages subpath, anywhere
  base: './',
  plugins: [react()],
})
