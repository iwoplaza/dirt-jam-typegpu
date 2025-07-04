<script lang="ts">
  import { game } from './lib/game';

  let canvas: HTMLCanvasElement;

  $effect(() => {
    const abortCtrl = new AbortController();
    game(canvas, abortCtrl.signal);
    return () => {
      abortCtrl.abort();
    };
  });
</script>

<main>
  <canvas bind:this={canvas}></canvas>

  <footer>
    Iwo Plaza © 2025
  </footer>
</main>

<style>
  canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
  }
</style>
