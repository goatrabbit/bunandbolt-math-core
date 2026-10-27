import { describe, expect, it } from 'vitest';
import {
  ALGORITHM_VERSION, bernoulli, bimodal, cauchy, die, drawnSampler, expo, gauss, laplace, mean, median, mulberry32, normal,
  sd, skewness, uniform,
} from '../src/lib/math/rng';
import { normalQuantile } from '../src/lib/math/special';

// The generator and every sampler must reproduce review/artifacts/page-directions/H19-sim-core.mjs draw for draw (its
// drawn-density inversion was changed with this library, 2026-10-07, before any page used it): a
// share link stores (seed, ALGORITHM_VERSION) and the page must replay the same samples. The first draws below were
// printed by H19-sim-core.mjs itself (node, 2026-10-07); the seeded totals are the ones H12, H19, H20 and H21 print.

describe('mulberry32 and the samplers: the same stream as H19-sim-core.mjs', () => {
  it('uniform draws', () => {
    const r = mulberry32(1);
    expect([r(), r(), r(), r(), r()]).toEqual([
      0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849,
    ]);
    expect(ALGORITHM_VERSION).toBe('stats-rng-v1');
  });
  it('Box–Muller (cos branch, one normal per two uniforms)', () => {
    const r = mulberry32(42);
    expect([gauss(r), gauss(r), gauss(r), gauss(r), gauss(r)]).toEqual([
      -0.956162229384149, -0.2730261048826104, -1.8416271847835968, -1.1408820709540242, -0.5294337937164153,
    ]);
  });
  it('every sampler in a fixed order on one stream', () => {
    const r = mulberry32(7);
    expect([die(r), bernoulli(r, 0.3), expo(r), laplace(r, 2), cauchy(r), uniform(r, -1, 3), bimodal(r, -2, 2, 0.5)]).toEqual([
      1, 1, 3.768253139403831, 0.927511386597965, -0.3058470282904662, 0.8649305300787091, -2.068780866451738,
    ]);
  });
  it('a drawn (piecewise-linear) density', () => {
    const s = drawnSampler([0, 1, 2, 3], [1, 3, 0, 2])!;
    const r = mulberry32(9);
    expect([s(r), s(r), s(r), s(r), s(r)]).toEqual([
      0.5697103025775763, 2.5748465736473527, 0.44230965196917654, 2.4444352911340683, 1.5976069934411978,
    ]);
  });
  it('a drawn density at any scale (Codex #18): only its shape matters', () => {
    expect(drawnSampler([0, 1], [0, 1e-13])!(() => 0.5)).toBe(Math.SQRT1_2); // density ∝ x: the median is √½
    const a = drawnSampler([0, 1, 2, 3], [1, 3, 0, 2])!, b = drawnSampler([0, 1, 2, 3], [1e-200, 3e-200, 0, 2e-200])!;
    for (const u of [0.01, 0.3, 0.5, 0.77, 0.999]) expect(Math.abs(a(() => u) - b(() => u))).toBeLessThan(1e-14);
  });
});

describe('seeded experiments reproduce the scripts', () => {
  it('H12: seed 1, one fair coin — 53 heads in 100, 50 103 in 100 000', () => {
    const r = mulberry32(1);
    let h = 0, at100 = 0;
    for (let n = 1; n <= 100000; n++) {
      h += bernoulli(r, 0.5);
      if (n === 100) at100 = h;
    }
    expect(at100).toBe(53);
    expect(h).toBe(50103);
  });
  it('H20: seed 1, n = 25 from N(50, 10), sample-major — 95% intervals cover 93/100 and 955/1000', () => {
    const r = mulberry32(1);
    const half = normalQuantile(0.975) * (10 / 5);
    let first100 = 0, first1000 = 0;
    for (let i = 0; i < 1000; i++) {
      const x: number[] = [];
      for (let j = 0; j < 25; j++) x.push(normal(r, 50, 10));
      const covered = Math.abs(mean(x) - 50) <= half ? 1 : 0;
      if (i < 100) first100 += covered;
      first1000 += covered;
    }
    expect(first100).toBe(93);
    expect(first1000).toBe(955);
  });
  it('H21: seed 1, 10 000 studies of n = 25 under H0 (μ = 500, σ = 10) — 225 with x̄ ≤ 496', () => {
    const r = mulberry32(1);
    let extreme = 0;
    for (let i = 0; i < 10000; i++) {
      let s = 0;
      for (let j = 0; j < 25; j++) s += normal(r, 500, 10);
      if (s / 25 <= 496) extreme++;
    }
    expect(extreme).toBe(225);
  });
  it('H19: seed 6, 10 000 means of 30 exponentials — SD 0.1826, mean 1.0033, skewness 0.338', () => {
    const r = mulberry32(6);
    const ms: number[] = [];
    for (let i = 0; i < 10000; i++) {
      let s = 0;
      for (let j = 0; j < 30; j++) s += expo(r);
      ms.push(s / 30);
    }
    expect(sd(ms).toFixed(4)).toBe('0.1826');
    expect(mean(ms).toFixed(4)).toBe('1.0033');
    expect(skewness(ms).toFixed(3)).toBe('0.338');
  });
});

describe('statistics helpers', () => {
  it('mean, sd (n − 1 by default), skewness, median', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(sd([2, 4, 4, 4, 5, 5, 7, 9], 0)).toBe(2);
    expect(sd([1, 2, 3, 4])).toBeCloseTo(Math.sqrt(5 / 3), 15);
    expect(skewness([1, 2, 3])).toBe(0);
    expect(skewness([0, 0, 0, 1])).toBeCloseTo(1.1547005383792515, 14); // (1/4·27/64)/(3/16)^1.5
    expect(median([5, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });
  it('samplers have the right moments (seed 3, 200 000 draws, within 5 standard errors)', () => {
    const r = mulberry32(3);
    const N = 200000;
    const check = (draw: () => number, mu: number, sigma: number) => {
      const xs = Array.from({ length: N }, draw);
      expect(Math.abs(mean(xs) - mu)).toBeLessThan((5 * sigma) / Math.sqrt(N));
    };
    check(() => gauss(r), 0, 1);
    check(() => expo(r), 1, 1);
    check(() => laplace(r, 2), 0, 2 * Math.SQRT2);
    check(() => die(r), 3.5, Math.sqrt(35 / 12));
    check(() => uniform(r, -1, 3), 1, 4 / Math.sqrt(12));
  });
});

describe('Codex re-review 2026-10-07: what the drawn sampler refuses', () => {
  it('#18 a bad grid is the page\u2019s bug: a RangeError (subnormal gap, decreasing, too short, not finite)', () => {
    expect(() => drawnSampler([0, 1e-310], [0, 1])).toThrow(RangeError);
    expect(() => drawnSampler([0, 2, 1], [1, 1, 1])).toThrow(RangeError);
    expect(() => drawnSampler([0], [1])).toThrow(RangeError);
    expect(() => drawnSampler([0, 1], [1])).toThrow(RangeError);
    expect(() => drawnSampler([0, 1], [1, NaN])).toThrow(RangeError);
    const v = drawnSampler([0, 1e-300], [0, 1])!(() => 0.5); // the smallest gap allowed: the median of a ramp, √½ of it
    expect(Math.abs(v / (Math.SQRT1_2 * 1e-300) - 1)).toBeLessThan(1e-14);
  });
  it('what a reader can draw never throws: no density (all 0, or below the axis) returns null', () => {
    expect(drawnSampler([0, 1, 2], [0, 0, 0])).toBeNull();
    expect(drawnSampler([0, 1], [1, -1])).toBeNull();
    expect(drawnSampler([0, 1, 2], [-0.5, -1, -2])).toBeNull();
    expect(drawnSampler([0, 1, 2], [0, 0, 1e-300])).not.toBeNull(); // a sliver of density is still a density
  });});
