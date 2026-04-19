import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import glsl from 'vite-plugin-glsl';

export default defineConfig({
  plugins: [react(), glsl()],
  server: {
    port: 5173,
    proxy: {
      '/patterns': 'http://127.0.0.1:8765',
      '/sim': 'http://127.0.0.1:8765',
      '/data': 'http://127.0.0.1:8765',
    },
  },
});
