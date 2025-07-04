import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { mat4 } from 'wgpu-matrix';

export async function game(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const root = await tgpu.init();

  const uniforms = root.createUniform(
    d.struct({
      proj: d.mat4x4f,
    }),
  );

  function resizeCanvas(canvas: HTMLCanvasElement) {
    const devicePixelRatio = window.devicePixelRatio;
    const width = window.innerWidth * devicePixelRatio;
    const height = window.innerHeight * devicePixelRatio;
    canvas.width = width;
    canvas.height = height;

    const aspect = canvas.width / canvas.height;
    const proj = mat4.perspective(Math.PI / 2, aspect, 0.01, 100, d.mat4x4f());
    uniforms.writePartial({ proj });
  }

  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);
  signal.addEventListener('abort', () => {
    resizeObserver.disconnect();
  });

  const mainVertex = tgpu['~unstable'].vertexFn({
    in: { idx: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
  })(({ idx }) => {
    const pos = [d.vec2f(0.0, 0.5), d.vec2f(-0.5, -0.5), d.vec2f(0.5, -0.5)];
    const uv = [d.vec2f(0.5, 1.0), d.vec2f(0.0, 0.0), d.vec2f(1.0, 0.0)];

    return { pos: d.vec4f(pos[idx], 0.0, 1.0), uv: uv[idx] };
  });

  const mainFragment = tgpu['~unstable'].fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(() => d.vec4f(1, 0, 0, 1));

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  const context = canvas.getContext('webgpu') as GPUCanvasContext;

  context.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  const pipeline = root['~unstable']
    .withVertex(mainVertex, {})
    .withFragment(mainFragment, { format: presentationFormat })
    .createPipeline();

  let animationFrame: number;
  function run() {
    pipeline
      .withColorAttachment({
        view: context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      })
      .draw(3);
    animationFrame = requestAnimationFrame(run);
  }
  requestAnimationFrame(run);

  signal.addEventListener('abort', () => {
    cancelAnimationFrame(animationFrame);
    root.destroy();
  });

  await root.device.queue.onSubmittedWorkDone();
}
