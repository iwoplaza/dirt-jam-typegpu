<script lang="ts">
  import { game } from './lib/game';

  let noWebGPUSupport = $state(false);

  let canvas: HTMLCanvasElement;

  $effect(() => {
    const abortCtrl = new AbortController();
    try {
      game(canvas, abortCtrl.signal);
    } catch (e) {
      noWebGPUSupport = true;
    }
    return () => {
      abortCtrl.abort();
    };
  });
</script>

<main>
  <canvas bind:this={canvas}></canvas>
  {#if noWebGPUSupport}
    <p>WebGPU seems to be not supported in your browser :(</p>
  {/if}

  <footer>
    <a href="https://github.com/iwoplaza" target="_blank" rel="noopener noreferrer">by Iwo Plaza</a>
  </footer>
</main>

<style>
  canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  footer {
    position: fixed;
    bottom: 1em;
    z-index: 100;
  }
</style>
