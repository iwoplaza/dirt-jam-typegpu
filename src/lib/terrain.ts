import tgpu from 'typegpu';
import { vec2f, f32, mat2x2f, u32, struct } from 'typegpu/data';
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
  /** How many layers of noise to sum. More octaves give more detail with diminishing returns (1, 32) */
  octaves: 16,
  /** Amount of layers used for computing the "smooth" gradient, used for material blending */
  smoothOctaves: 6,
  /** Value to multiply with amplitude each octave iteration, lower values will reduce the impact of each subsequent octave (0.01, 1.0) */
  amplitudeDecay: 0.45,
  offset: vec2f(12.21, 9.2),
});

const FbmResult = struct({
  grad: vec2f,
  smoothGrad: vec2f,
  height: f32,
});

/**
 * The fractional brownian motion that sums many noise values as explained in the video accompanying this project
 */
export const fbm = tgpu.fn(
  [vec2f],
  FbmResult,
)((pos) => {
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

  for (let i = u32(0); i < u32(settingsSlot.$.octaves); i++) {
    const n = mul(amplitude, perlin2d.sampleWithGradient(tPos));

    // add height scaled by current amplitude
    height += n.x;

    // add gradient scaled by amplitude and transformed by accumulated rotations
    grad = add(grad, mul(m, n.yz));
    if (i === settingsSlot.$.smoothOctaves) {
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
