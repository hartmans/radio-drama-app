import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    hmr: false,
  },
  preview: {
    port: 3000,  // Use simple port for testing
    host: true,
  },
})
