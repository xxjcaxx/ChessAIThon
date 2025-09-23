import { defineConfig } from 'vite'

export default defineConfig({
    optimizeDeps: {
    exclude: ['chessmarro-board'] // <- muy importante
  },
  resolve: {
    preserveSymlinks: true, 
  }
})
