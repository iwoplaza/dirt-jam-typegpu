import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { perlin2d } from '@typegpu/noise';
import { mat4 } from 'wgpu-matrix';

export async function game(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const root = await tgpu.init();

  const uniforms = root.createUniform(
    d.struct({
      viewProj: d.mat4x4f,
    }),
  );

  function resizeCanvas(canvas: HTMLCanvasElement) {
    const devicePixelRatio = window.devicePixelRatio;
    const width = window.innerWidth * devicePixelRatio;
    const height = window.innerHeight * devicePixelRatio;
    canvas.width = width;
    canvas.height = height;

    const aspect = canvas.width / canvas.height;
    const proj = mat4.perspective(
      (50 / 180) * Math.PI,
      aspect,
      0.01,
      1000,
      d.mat4x4f(),
    );
    const dist = 10;
    const view = mat4.lookAt(
      std.mul(dist, d.vec3f(-5, 4, -5)),
      std.mul(dist, d.vec3f(10, 0, 10)),
      d.vec3f(0, 1, 0),
      d.mat4x4f(),
    );
    const viewProj = mat4.mul(proj, view, d.mat4x4f());
    uniforms.writePartial({ viewProj });
  }

  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);

  const TILES_X = 1024;
  const TILES_Z = 1024;
  const TERRAIN_FREQ = 0.02;
  const TERRAIN_PERIOD = 1 / TERRAIN_FREQ;
  const TERRAIN_HEIGHT = 10;

  const Varying = {
    uv: d.vec2f,
    samplePos: d.vec2f,
  };

  const mainVertex = tgpu['~unstable'].vertexFn({
    in: { idx: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, ...Varying },
  })(({ idx }) => {
    const localIdx = idx % 6;
    const tileIdx = idx / 6;
    const tileX = d.f32(tileIdx % TILES_X);
    const tileZ = d.f32(tileIdx / TILES_X);

    const localOffset = [
      d.vec3f(0, 0, 0),
      d.vec3f(0, 0, 1),
      d.vec3f(1, 0, 1),

      d.vec3f(0, 0, 0),
      d.vec3f(1, 0, 1),
      d.vec3f(1, 0, 0),
    ];
    const origin = d.vec3f(tileX, 0, tileZ);
    const localPos = std.add(origin, localOffset[localIdx]);
    const samplePos = std.mul(localPos.xz, TERRAIN_FREQ);
    const worldPos = d.vec3f(
      localPos.x,
      localPos.y + perlin2d.sample(samplePos) * TERRAIN_HEIGHT,
      localPos.z,
    );

    return {
      pos: std.mul(uniforms.$.viewProj, d.vec4f(worldPos, 1)),
      samplePos,
      uv: localOffset[localIdx].xz,
    };
  });

  const lightDir = std.normalize(d.vec3f(1, 1, 1));
  const mainFragment = tgpu['~unstable'].fragmentFn({
    in: { ...Varying },
    out: d.vec4f,
  })(({ uv, samplePos }) => {
    const noise = std.mul(
      TERRAIN_HEIGHT,
      perlin2d.sampleWithGradient(samplePos),
    );
    const normal = std.normalize(d.vec3f(-noise.y, TERRAIN_PERIOD, -noise.z));
    const att = std.max(0, std.dot(lightDir, normal));

    const shaderColor = d.vec3f(0.2, 0.2, 0.4);
    const litColor = d.vec3f(1, 1, 1);
    return d.vec4f(std.mix(shaderColor, litColor, att), 1);
  });

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
        clearValue: [1, 1, 1, 1],
        loadOp: 'clear',
        storeOp: 'store',
      })
      .draw(6 * TILES_X * TILES_Z);

    animationFrame = requestAnimationFrame(run);
  }
  requestAnimationFrame(run);

  signal.addEventListener('abort', () => {
    cancelAnimationFrame(animationFrame);
    resizeObserver.disconnect();
    root.destroy();
  });

  await root.device.queue.onSubmittedWorkDone();
}
