import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { mat4 } from 'wgpu-matrix';
import { fbm } from './terrain';
import { perlin2d } from '@typegpu/noise';

export async function game(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const root = await tgpu.init();
  const device = root.device;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const perlinCache = perlin2d.staticCache({ root, size: d.vec2u(16) });

  const uniforms = root.createUniform(
    d.struct({
      viewProj: d.mat4x4f,
    }),
  );

  // Textures

  let depthTexture: GPUTexture;
  let depthTextureView: GPUTextureView;
  let msaaTexture: GPUTexture;
  let msaaTextureView: GPUTextureView;

  function createDepthAndMsaaTextures() {
    if (depthTexture) {
      depthTexture.destroy();
    }
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: 'depth24plus',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthTextureView = depthTexture.createView();

    if (msaaTexture) {
      msaaTexture.destroy();
    }
    msaaTexture = device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: presentationFormat,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    msaaTextureView = msaaTexture.createView();
  }
  createDepthAndMsaaTextures();

  let proj = mat4.identity(d.mat4x4f());
  let view = mat4.identity(d.mat4x4f());

  function uploadUniforms() {
    const viewProj = mat4.mul(proj, view, d.mat4x4f());
    uniforms.writePartial({ viewProj });
  }

  function resizeCanvas(canvas: HTMLCanvasElement) {
    const devicePixelRatio = window.devicePixelRatio;
    const width = window.innerWidth * devicePixelRatio;
    const height = window.innerHeight * devicePixelRatio;
    canvas.width = width;
    canvas.height = height;

    const aspect = canvas.width / canvas.height;
    proj = mat4.perspective(
      (50 / 180) * Math.PI,
      aspect,
      0.01,
      1000,
      d.mat4x4f(),
    );
    uploadUniforms();
    createDepthAndMsaaTextures();
  }

  resizeCanvas(canvas);
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);

  let isDragging = false;
  let prevX = 0;
  let prevY = 0;
  // Yaw and pitch angles facing the origin.
  // let orbitRadius = 30;
  // let orbitYaw = 0.09;
  // let orbitPitch = 0.005;
  let orbitRadius = 23.6;
  let orbitYaw = -12.78;
  let orbitPitch = 0.01;

  function updateCameraOrbit(dx: number, dy: number) {
    console.log({
      orbitRadius,
      orbitYaw,
      orbitPitch,
    });

    const orbitSensitivity = 0.005;
    orbitYaw += -dx * orbitSensitivity;
    orbitPitch += dy * orbitSensitivity;
    // if we didn't limit pitch, it would lead to flipping the camera which is disorienting.
    const maxPitch = Math.PI / 2 - 0.01;
    if (orbitPitch > maxPitch) orbitPitch = maxPitch;
    if (orbitPitch < -maxPitch) orbitPitch = -maxPitch;
    // basically converting spherical coordinates to cartesian.
    // like sampling points on a unit sphere and then scaling them by the radius.
    const logOrbitRadius = orbitRadius ** 2;
    const newCamX = logOrbitRadius * Math.sin(orbitYaw) * Math.cos(orbitPitch);
    const newCamY = logOrbitRadius * Math.sin(orbitPitch);
    const newCamZ = logOrbitRadius * Math.cos(orbitYaw) * Math.cos(orbitPitch);
    const newCameraPos = d.vec4f(newCamX, newCamY, newCamZ, 1);

    view = mat4.lookAt(newCameraPos, d.vec3f(0), d.vec3f(0, 1, 0), d.mat4x4f());
    const viewProj = mat4.mul(proj, view, d.mat4x4f());
    uniforms.writePartial({ viewProj });
  }

  canvas.addEventListener('wheel', (event: WheelEvent) => {
    event.preventDefault();
    const zoomSensitivity = 0.005;
    orbitRadius = Math.max(1, orbitRadius + event.deltaY * zoomSensitivity);
    updateCameraOrbit(0, 0);
  });

  canvas.addEventListener('mousedown', (event) => {
    if (event.button === 0) {
      // Left Mouse Button controls Camera Orbit.
      isDragging = true;
    }
    prevX = event.clientX;
    prevY = event.clientY;
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  canvas.addEventListener('mousemove', (event) => {
    const dx = event.clientX - prevX;
    const dy = event.clientY - prevY;
    prevX = event.clientX;
    prevY = event.clientY;

    if (isDragging) {
      updateCameraOrbit(dx, dy);
    }
  });

  // Mobile touch support.
  canvas.addEventListener('touchstart', (event: TouchEvent) => {
    event.preventDefault();
    if (event.touches.length === 1) {
      // Single touch controls Camera Orbit.
      isDragging = true;
    }
    // Use the first touch for rotation.
    prevX = event.touches[0].clientX;
    prevY = event.touches[0].clientY;
  });

  canvas.addEventListener('touchmove', (event: TouchEvent) => {
    event.preventDefault();
    const touch = event.touches[0];
    const dx = touch.clientX - prevX;
    const dy = touch.clientY - prevY;
    prevX = touch.clientX;
    prevY = touch.clientY;

    if (isDragging && event.touches.length === 1) {
      updateCameraOrbit(dx, dy);
    }
  });

  canvas.addEventListener('touchend', (event: TouchEvent) => {
    event.preventDefault();
    if (event.touches.length === 0) {
      isDragging = false;
    }
  });

  const TILES_X = 2048;
  const TILES_Z = 2048;
  const _Scale = 300;

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
    const origin = d.vec3f(tileX - TILES_X / 2, 0, tileZ - TILES_Z / 2);
    const localPos = std.add(origin, localOffset[localIdx]);
    const samplePos = std.div(localPos.xz, _Scale);
    const noise = fbm(samplePos);
    const worldPos = d.vec3f(
      localPos.x,
      localPos.y + noise.x * _Scale,
      localPos.z,
    );

    return {
      pos: std.mul(uniforms.$.viewProj, d.vec4f(worldPos, 1)),
      samplePos,
      uv: localOffset[localIdx].xz,
    };
  });

  const fogStart = 0;
  const fogEnd = 600;
  const lightDir = std.normalize(d.vec3f(1, 0.5, -1));
  const fogColor = d.vec3f(1, 1, 1);

  const mainFragment = tgpu['~unstable'].fragmentFn({
    in: { ...Varying, pixel: d.builtin.position },
    out: d.vec4f,
  })(({ uv, samplePos, pixel }) => {
    const noise = fbm(samplePos);
    const normal = std.normalize(d.vec3f(-noise.y, 1, -noise.z));
    const att = std.max(0, std.dot(lightDir, normal));

    const shadowColor = d.vec3f(0);
    const albedo = d.vec3f(0.3, 0.28, 0.22);
    const fog = std.pow(
      std.clamp(
        (pixel.z / pixel.w - fogStart) / d.f32(fogEnd - fogStart),
        0,
        1,
      ),
      0.5,
    );

    const terrainColor = std.mix(shadowColor, albedo, att);
    const fogified = std.mix(terrainColor, fogColor, fog);
    return d.vec4f(fogified, 1);
  });

  const context = canvas.getContext('webgpu') as GPUCanvasContext;

  context.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  const pipeline = root['~unstable']
    .with(perlin2d.getJunctionGradientSlot, perlinCache.getJunctionGradient)
    .withVertex(mainVertex, {})
    .withFragment(mainFragment, { format: presentationFormat })
    .withDepthStencil({
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    })
    .withMultisample({
      count: 4,
    })
    .withPrimitive({
      cullMode: 'back',
    })
    .createPipeline();

  let animationFrame: number;
  function run() {
    pipeline
      .withColorAttachment({
        view: msaaTextureView,
        resolveTarget: context.getCurrentTexture().createView(),
        clearValue: [1, 1, 1, 1],
        loadOp: 'clear',
        storeOp: 'store',
      })
      .withDepthStencilAttachment({
        view: depthTextureView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
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

  updateCameraOrbit(0, 0);
}
