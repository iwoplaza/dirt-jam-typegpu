import tgpu from 'typegpu';
import { vec2f, f32, mat2x2f, u32, struct, vec3f } from 'typegpu/data';
import { add, cos, mul, sin } from 'typegpu/std';
import { perlin2d } from '@typegpu/noise';

/**
 * Can be swapped with a different value at compile time, but
 * the default are fiiiine.
 */
const settingsSlot = tgpu.slot({
  /** Self similarity of each octave (0.01, 3.0) */
  lacunarity: 2,
  /** Amplitude of the first noise octave (0.01, 2.0) */
  baseAmplitude: 0.4,
  /** Amount of rotation (in radians) to apply each octave iteration */
  noiseRotation: (30 / 180) * Math.PI,
  /** Value to multiply with amplitude each octave iteration, lower values will reduce the impact of each subsequent octave (0.01, 1.0) */
  amplitudeDecay: 0.45,
  offset: vec2f(12.21, 9.2),
});

/**
 * The fractional brownian motion that sums many noise values as explained in the video accompanying this project
 */
export const fbm = tgpu.fn(
  [vec2f, u32],
  f32,
)((pos, octaves) => {
  const lacunarity = f32(settingsSlot.$.lacunarity);
  const theta = f32(settingsSlot.$.noiseRotation);
  let tPos = add(pos, settingsSlot.$.offset);
  let amplitude = f32(settingsSlot.$.baseAmplitude);

  // height sum
  let height = f32(0);

  // accumulated rotations

  // rotation matrix
  let m2 = mat2x2f(
    vec2f(cos(theta), -sin(theta)),
    vec2f(sin(theta), cos(theta)),
  );

  // inverse rotation matrix
  let m2i = mat2x2f(
    vec2f(cos(theta), sin(theta)),
    vec2f(-sin(theta), cos(theta)),
  );

  for (let i = u32(0); i < octaves; i++) {
    const n = mul(amplitude, perlin2d.sampleWithGradient(tPos));

    // add height scaled by current amplitude
    height += n.x;

    // apply amplitude decay to reduce impact of next noise layer
    amplitude *= settingsSlot.$.amplitudeDecay;

    // reconstruct rotation matrix, kind of a performance stink since this is technically expensive and doesn't need to be done if no random angle variance but whatever it's 2025
    m2 = mat2x2f(vec2f(cos(theta), -sin(theta)), vec2f(sin(theta), cos(theta)));

    // inverse rotation matrix
    m2i = mat2x2f(
      vec2f(cos(theta), sin(theta)),
      vec2f(-sin(theta), cos(theta)),
    );

    // apply frequency adjustment to sample position for next noise layer
    tPos = mul(lacunarity, mul(m2, tPos));
  }

  return height;
});

/**
 * The fractional brownian motion that sums many noise values as explained in the video accompanying this project
 */
export const fbmWithGradient = tgpu.fn(
  [vec2f, u32],
  vec3f,
)((pos, octaves) => {
  const lacunarity = f32(settingsSlot.$.lacunarity);
  const theta = f32(settingsSlot.$.noiseRotation);
  let tPos = add(pos, settingsSlot.$.offset);
  let amplitude = f32(settingsSlot.$.baseAmplitude);

  // height sum
  let height = f32(0);

  // derivative sums
  let grad = vec2f(0);

  // accumulated rotations
  let m = mat2x2f(1, 0, 0, 1);

  // rotation matrix
  let m2 = mat2x2f(
    vec2f(cos(theta), -sin(theta)),
    vec2f(sin(theta), cos(theta)),
  );

  // inverse rotation matrix
  let m2i = mat2x2f(
    vec2f(cos(theta), sin(theta)),
    vec2f(-sin(theta), cos(theta)),
  );

  for (let i = u32(0); i < octaves; i++) {
    const n = mul(amplitude, perlin2d.sampleWithGradient(tPos));

    // add height scaled by current amplitude
    height += n.x;

    // add gradient scaled by amplitude and transformed by accumulated rotations
    grad = add(grad, mul(m, n.yz));

    // apply amplitude decay to reduce impact of next noise layer
    amplitude *= settingsSlot.$.amplitudeDecay;

    // reconstruct rotation matrix, kind of a performance stink since this is technically expensive and doesn't need to be done if no random angle variance but whatever it's 2025
    m2 = mat2x2f(vec2f(cos(theta), -sin(theta)), vec2f(sin(theta), cos(theta)));

    // inverse rotation matrix
    m2i = mat2x2f(
      vec2f(cos(theta), sin(theta)),
      vec2f(-sin(theta), cos(theta)),
    );

    // apply frequency adjustment to sample position for next noise layer
    tPos = mul(lacunarity, mul(m2, tPos));
    m = mul(lacunarity, mul(m2i, m));
  }

  return vec3f(height, grad);
});

const FbmDoubleResult = struct({
  grad: vec2f,
  smoothGrad: vec2f,
  height: f32,
});

/**
 * The fractional brownian motion that sums many noise values as explained in the video accompanying this project
 */
export const fbmWithDoubleGradient = tgpu.fn(
  [vec2f, u32, u32],
  FbmDoubleResult,
)((pos, firstOctaveThreshold, secondOctaveThreshold) => {
  const lacunarity = f32(settingsSlot.$.lacunarity);
  const theta = f32(settingsSlot.$.noiseRotation);
  let tPos = add(pos, settingsSlot.$.offset);
  let amplitude = f32(settingsSlot.$.baseAmplitude);

  // height sum
  let height = f32(0);

  // derivative sums
  let grad = vec2f(0);
  let smoothGrad = vec2f(0);

  // accumulated rotations
  let m = mat2x2f(1, 0, 0, 1);

  // rotation matrix
  let m2 = mat2x2f(
    vec2f(cos(theta), -sin(theta)),
    vec2f(sin(theta), cos(theta)),
  );

  // inverse rotation matrix
  let m2i = mat2x2f(
    vec2f(cos(theta), sin(theta)),
    vec2f(-sin(theta), cos(theta)),
  );

  for (let i = u32(0); i < secondOctaveThreshold; i++) {
    const n = mul(amplitude, perlin2d.sampleWithGradient(tPos));

    // add height scaled by current amplitude
    height += n.x;

    // add gradient scaled by amplitude and transformed by accumulated rotations
    grad = add(grad, mul(m, n.yz));
    if (i === firstOctaveThreshold) {
      smoothGrad = grad;
    }

    // apply amplitude decay to reduce impact of next noise layer
    amplitude *= settingsSlot.$.amplitudeDecay;

    // reconstruct rotation matrix, kind of a performance stink since this is technically expensive and doesn't need to be done if no random angle variance but whatever it's 2025
    m2 = mat2x2f(vec2f(cos(theta), -sin(theta)), vec2f(sin(theta), cos(theta)));

    // inverse rotation matrix
    m2i = mat2x2f(
      vec2f(cos(theta), sin(theta)),
      vec2f(-sin(theta), cos(theta)),
    );

    // apply frequency adjustment to sample position for next noise layer
    tPos = mul(lacunarity, mul(m2, tPos));
    m = mul(lacunarity, mul(m2i, m));
  }

  return {
    grad,
    smoothGrad,
    height,
  };
});
