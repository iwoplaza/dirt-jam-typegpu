import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import typegpu from 'unplugin-typegpu/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte(), typegpu({})],
});
