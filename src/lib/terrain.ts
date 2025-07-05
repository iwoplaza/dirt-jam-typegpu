import tgpu from 'typegpu';
import { vec2f, vec3f, f32, mat2x2f, u32 } from 'typegpu/data';
import { add, cos, mul, sin } from 'typegpu/std';
import { perlin2d, randf } from '@typegpu/noise';

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
  octaves: 10,
  /** Value to multiply with amplitude each octave iteration, lower values will reduce the impact of each subsequent octave (0.01, 1.0) */
  amplitudeDecay: 0.45,
});

/**
 * The fractional brownian motion that sums many noise values as explained in the video accompanying this project
 */
export const fbm = tgpu.fn(
  [vec2f],
  vec3f,
)((pos) => {
  let tPos = vec2f(pos);
  const lacunarity = f32(settingsSlot.$.lacunarity);
  let amplitude = f32(settingsSlot.$.baseAmplitude);

  // height sum
  let height = f32(0);

  // derivative sum
  let grad = vec2f(0);

  // accumulated rotations
  let m = mat2x2f(1, 0, 0, 1);

  randf.seed(123);

  const theta = settingsSlot.$.noiseRotation;

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
    grad = add(grad, mul(amplitude, mul(m, n.yz)));

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
