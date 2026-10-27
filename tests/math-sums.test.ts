import { describe, expect, it } from 'vitest';
import {
  IRWIN_HALL_MAX, binomialBand, binomialPmf, distanceToNormal, latticeSumLowerQuantiles, inversionCheck, inversionGridSize, irwinHallCdf, irwinHallPdf, latticeSumPmf,
  meanDensityByInversion, meanOfTwoNormals, mixtureCdf, mixturePdf, polylineCf, polylineMoments, symmetricTrinomialPmf, uniformCf,
} from '../src/lib/math/sums';
import { gammaCdf, normalCdf } from '../src/lib/math/special';
import { gaussLegendre } from '../src/lib/math/quad';
import { bimodal, mulberry32 } from '../src/lib/math/rng';
import { ddLowerWalk } from './dd-band';

// The distributions of sums and means behind the central limit theorem page (design/pages/H19-central-limit-theorem.md
// 数学). Expected values marked "H19.mjs" are the ones review/artifacts/page-directions/H19.mjs prints and checks.

const total = (p: ArrayLike<number>) => { let s = 0; for (let i = 0; i < p.length; i++) s += p[i]; return s; };
const DIE = Array(6).fill(1 / 6);

describe('lattice sums: the die, exact by adding one copy at a time', () => {
  it('two and three dice (H19.mjs §1)', () => {
    const two = latticeSumPmf(DIE, 2), three = latticeSumPmf(DIE, 3); // index = sum − n
    expect(two).toHaveLength(11);
    expect(two[7 - 2]).toBeCloseTo(6 / 36, 15);
    expect(three[10 - 3]).toBeCloseTo(27 / 216, 15);
    expect(three[11 - 3]).toBeCloseTo(27 / 216, 15);
    expect(27 / 216).toBe(0.125);
    // every count: (ways to make s) / 6ⁿ
    expect([...two].map((p) => Math.round(p * 36))).toEqual([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]);
  });
  it('mean and variance of the sum: n·3.5 and n·35/12', () => {
    for (const n of [1, 10, 200]) {
      const p = latticeSumPmf(DIE, n);
      let m = 0, v = 0;
      p.forEach((q, k) => { m += q * (k + n); });
      p.forEach((q, k) => { v += q * (k + n - m) ** 2; });
      expect(m).toBeCloseTo(3.5 * n, 9);
      expect(v / (35 / 12 * n)).toBeCloseTo(1, 11);
    }
  });
  it('sums to 1 (1e-12) and stays non-negative, up to n = 3000', () => {
    for (const n of [1, 2, 7, 100, 1000, 3000]) {
      const p = latticeSumPmf(DIE, n);
      expect(Math.abs(total(p) - 1)).toBeLessThan(1e-12);
      expect(p.every((q) => q >= 0)).toBe(true);
    }
  });
  it('a base with zeros, and the edge cases', () => {
    const rare = latticeSumPmf([0.99, 0.01], 30), bin = binomialPmf(30, 0.01);
    rare.forEach((q, k) => expect(q).toBeCloseTo(bin[k], 15));
    expect([...latticeSumPmf([0, 0.5, 0.5], 2)]).toEqual([0, 0, 0.25, 0.5, 0.25]);
    expect([...latticeSumPmf(DIE, 0)]).toEqual([1]);
    expect(() => latticeSumPmf(DIE, -1)).toThrow(RangeError);
    expect(() => latticeSumPmf(DIE, 2.5)).toThrow(RangeError);
  });
});

describe('the two rare events, closed forms', () => {
  it('Bernoulli(0.01): P(mean = 0) = 0.99ⁿ (H19.mjs §3)', () => {
    expect(binomialPmf(30, 0.01)[0] / 0.99 ** 30).toBeCloseTo(1, 14);
    expect(binomialPmf(30, 0.01)[0]).toBeCloseTo(0.7397, 4);
    expect(binomialPmf(1000, 0.01)[0] / 4.317124741065786e-5).toBeCloseTo(1, 12);
  });
  it('binomial pmfs sum to 1 (1e-12) at every size, and agree with C(n, k)pᵏqⁿ⁻ᵏ', () => {
    for (const [n, p] of [[30, 0.01], [1000, 0.01], [10000, 0.01], [10000, 0.5], [7, 0.3]] as const) expect(Math.abs(total(binomialPmf(n, p)) - 1)).toBeLessThan(1e-12);
    const b = binomialPmf(7, 0.3), C = [1, 7, 21, 35, 35, 21, 7, 1];
    b.forEach((q, k) => expect(q / (C[k] * 0.3 ** k * 0.7 ** (7 - k))).toBeCloseTo(1, 13));
    expect([...binomialPmf(3, 0)]).toEqual([1, 0, 0, 0]);
    expect([...binomialPmf(3, 1)]).toEqual([0, 0, 0, 1]);
  });
  it('symmetric rare event: P(mean = 0) at n = 30 is 0.7562, of which 0.7397 is "all 30 are 0" (H19.mjs §7)', () => {
    const p = symmetricTrinomialPmf(30, 0.005); // index d + n
    expect(p[30]).toBeCloseTo(0.7561948001711157, 12);
    expect(binomialPmf(30, 0.01)[0]).toBeCloseTo(0.7397003733882802, 14);
    for (let d = 1; d <= 30; d++) expect(p[30 + d]).toBeCloseTo(p[30 - d], 16); // symmetric
    let v = 0;
    p.forEach((q, i) => { v += q * (i - 30) ** 2; });
    expect(v).toBeCloseTo(30 * 0.01, 12); // Var D = n·2q (σ = 1 when the values are ±10: 100·0.01 = 1)
    for (const n of [1, 30, 1000, 10000]) expect(Math.abs(total(symmetricTrinomialPmf(n, 0.005)) - 1)).toBeLessThan(1e-12);
  });
});

describe('Irwin–Hall: the uniform up to n = 50', () => {
  it('the density integrates to 1, the CDF is ½ at the centre and its derivative is the density', () => {
    for (const n of [1, 2, 3, 10, 30, 50]) {
      // piecewise polynomial of degree n − 1 between integers: Gauss–Legendre on each unit is exact
      let s = 0;
      for (let k = 0; k < n; k++) s += gaussLegendre((x) => irwinHallPdf(x, n), k, k + 1, 32);
      // the alternating sum loses digits as n grows: about 12 left at n = 30, 9 at n = 50 (the design doc's claim)
      const tol = n <= 10 ? 1e-14 : n <= 30 ? 1e-11 : 1e-8;
      expect(Math.abs(s - 1)).toBeLessThan(tol);
      expect(Math.abs(irwinHallCdf(n / 2, n) - 0.5)).toBeLessThan(tol);
      const x = 0.37 * n, h = 1e-5;
      expect((irwinHallCdf(x + h, n) - irwinHallCdf(x - h, n)) / (2 * h)).toBeCloseTo(irwinHallPdf(x, n), 6);
    }
    expect(irwinHallPdf(0.5, 2)).toBeCloseTo(0.5, 15);
    expect(irwinHallPdf(1.5, 3)).toBeCloseTo(0.75, 15);
    expect(irwinHallPdf(-0.1, 3)).toBe(0);
    expect(irwinHallCdf(3.1, 3)).toBe(1);
    // far right of the middle the symmetry keeps it accurate (Codex design review: a direct alternating sum at
    // P(S₅₀ ≤ 35) gives 0.249; the 100-digit value is 0.999999731…)
    expect(irwinHallCdf(35, 50)).toBeCloseTo(0.999999731, 8);
  });
  it('refuses the n where the alternating sum is no longer trustworthy', () => {
    expect(IRWIN_HALL_MAX).toBe(50);
    expect(() => irwinHallPdf(1, 51)).toThrow(RangeError);
    expect(() => irwinHallCdf(1, 0)).toThrow(RangeError);
    expect(() => irwinHallPdf(1, 2.5)).toThrow(RangeError);
  });
  it('agrees with the numeric inversion at n = 30 and 50 (1e-6)', () => {
    for (const n of [30, 50]) {
      const M = inversionGridSize(n, 1 / Math.sqrt(12), 0, 1);
      const g = meanDensityByInversion(uniformCf(0, 1), 0, 1, n, M);
      let worst = 0;
      for (let j = 0; j < M; j++) { const x = g.x0 + j * g.dx; worst = Math.max(worst, Math.abs(n * irwinHallPdf(n * x, n) - g.f[j])); }
      expect(worst).toBeLessThan(1e-6);
    }
  });
});

describe('two humps: the mean is a finite normal mixture', () => {
  it('n = 4: weight 1, mean 0, variance 4.25/4', () => {
    const mx = meanOfTwoNormals(4, -2, 2, 0.5);
    expect(Math.abs(total(mx.w) - 1)).toBeLessThan(1e-14);
    expect(mx.s).toBe(0.25);
    const m1 = gaussLegendre((x) => x * mixturePdf(mx, x), -6, 6, 20, 60);
    const m2 = gaussLegendre((x) => x * x * mixturePdf(mx, x), -6, 6, 20, 60);
    expect(gaussLegendre((x) => mixturePdf(mx, x), -6, 6, 20, 60)).toBeCloseTo(1, 12);
    expect(m1).toBeCloseTo(0, 12);
    expect(m2).toBeCloseTo(4.25 / 4, 11);
    expect(Math.abs(mixtureCdf(mx, 0) - 0.5)).toBeLessThan(1e-14);
  });
  it('seed 8, 20 000 means of n = 4: the empirical CDF stays within the 1% Kolmogorov bound (H19.mjs §8)', () => {
    const mx = meanOfTwoNormals(4, -2, 2, 0.5), r = mulberry32(8), ms: number[] = [];
    for (let i = 0; i < 20000; i++) { let t = 0; for (let j = 0; j < 4; j++) t += bimodal(r, -2, 2, 0.5); ms.push(t / 4); }
    ms.sort((a, b) => a - b);
    let d = 0;
    ms.forEach((x, i) => { const F = mixtureCdf(mx, x); d = Math.max(d, Math.abs(F - (i + 1) / ms.length), Math.abs(F - i / ms.length)); });
    expect(d).toBeCloseTo(0.005961755560810689, 9);
    expect(d).toBeLessThan(1.63 / Math.sqrt(20000));
  });
  it('drops only components that are invisible (n = 10 000)', () => {
    const mx = meanOfTwoNormals(10000, -2, 2, 0.5);
    expect(mx.w.length).toBeLessThan(1001);
    expect(Math.abs(total(mx.w) - 1)).toBeLessThan(1e-14);
  });
});

describe('a drawn (piecewise-linear) density', () => {
  it('moments of a uniform and of a triangle, exactly', () => {
    const u = polylineMoments([0, 2], [1, 1]);
    expect(u.area).toBe(2);
    expect(u.mean).toBeCloseTo(1, 15);
    expect(u.sd).toBeCloseTo(1 / Math.sqrt(3), 15);
    expect(u.skew).toBeCloseTo(0, 14);
    expect(u.kurt).toBeCloseTo(-1.2, 14);
    // triangle (a, c, b) = (0, 1, 3): mean 4/3, variance 7/18, skewness √2·20/(5·7^{3/2}), excess kurtosis −3/5
    const t = polylineMoments([0, 1, 3], [0, 2, 0]);
    expect(t.area).toBeCloseTo(3, 15);
    expect(t.mean).toBeCloseTo(4 / 3, 15);
    expect(t.sd).toBeCloseTo(Math.sqrt(7 / 18), 15);
    expect(t.skew).toBeCloseTo((Math.SQRT2 * 20) / (5 * 7 ** 1.5), 14);
    expect(t.kurt).toBeCloseTo(-0.6, 14);
    expect(() => polylineMoments([0, 1], [0, 0])).toThrow(RangeError);
  });
  it('its characteristic function: 1 at 0, and the integral it stands for', () => {
    const x = [0, 1, 3, 3.5], h = [0.5, 2, 0.2, 1];
    const cf = polylineCf(x, h), A = polylineMoments(x, h).area;
    expect(cf(0)).toEqual([1, 0]);
    const f = (u: number) => { for (let i = 0; i + 1 < x.length; i++) if (u <= x[i + 1]) return h[i] + ((h[i + 1] - h[i]) * (u - x[i])) / (x[i + 1] - x[i]); return 0; };
    for (const t of [1e-7, 0.3, 1.9, 7, 50]) {
      let re = 0, im = 0;
      for (let i = 0; i + 1 < x.length; i++) {
        re += gaussLegendre((u) => f(u) * Math.cos(t * u), x[i], x[i + 1], 20, 40);
        im += gaussLegendre((u) => f(u) * Math.sin(t * u), x[i], x[i + 1], 20, 40);
      }
      const [cr, ci] = cf(t);
      expect(cr).toBeCloseTo(re / A, 12);
      expect(ci).toBeCloseTo(im / A, 12);
      expect(Math.hypot(cr, ci)).toBeLessThanOrEqual(cf.bound!(t) + 1e-15); // the bound holds
    }
  });
  it('the uniform characteristic function is (e^{itb} − e^{ita})/(it(b − a))', () => {
    const cf = uniformCf(-1, 3);
    for (const t of [0.4, 2, 9]) {
      const re = (Math.sin(3 * t) - Math.sin(-t)) / (4 * t), im = (-Math.cos(3 * t) + Math.cos(-t)) / (4 * t);
      expect(cf(t)[0]).toBeCloseTo(re, 14);
      expect(cf(t)[1]).toBeCloseTo(im, 14);
    }
    expect(cf(0)).toEqual([1, 0]);
  });
});

describe('the mean by Fourier inversion (numeric)', () => {
  it('two uniforms make the triangle 4x ∧ 4(1 − x): 1e-4 away from its three kinks at M = 4096', () => {
    const g = meanDensityByInversion(uniformCf(0, 1), 0, 1, 2, 4096);
    expect(g.x0).toBe(0);
    expect(g.dx).toBe(1 / 4096);
    let far = 0, near = 0;
    for (let j = 0; j < 4096; j++) {
      const x = j * g.dx, e = Math.abs((x < 0.5 ? 4 * x : 4 * (1 - x)) - g.f[j]);
      if (Math.min(x, Math.abs(x - 0.5), 1 - x) > 0.01) far = Math.max(far, e); else near = Math.max(near, e);
    }
    expect(far).toBeLessThan(1e-4);
    expect(near).toBeLessThan(5e-4); // at a kink the truncated series is off by O(1/M)
    expect(total(g.f) * g.dx).toBeCloseTo(1, 12);
  });
  const xs = Array.from({ length: 81 }, (_, i) => i / 8);
  const hs = xs.map((x) => Math.exp(-((x - 2) ** 2) / 1.5) + 0.35 * Math.exp(-((x - 7) ** 2) / 0.8)); // a lopsided drawing
  const mom = polylineMoments(xs, hs), cf = polylineCf(xs, hs);
  it.each([2, 30, 1000])('a lopsided drawing, n = %i: passes the page\'s check (< 1e-4), mean μ and sd σ/√n', (n) => {
    const M = inversionGridSize(n, mom.sd, 0, 10);
    expect(inversionCheck(cf, 0, 10, n, M)).toBeLessThan(1e-4);
    const g = meanDensityByInversion(cf, 0, 10, n, M);
    let m0 = 0, m1 = 0, m2 = 0;
    g.f.forEach((f, j) => { const x = g.x0 + j * g.dx; m0 += f; m1 += f * x; m2 += f * x * x; });
    m0 *= g.dx; m1 *= g.dx; m2 *= g.dx;
    expect(Math.abs(m0 - 1)).toBeLessThan(1e-7); // the few small negative values set to 0 add a little mass
    expect(m1 / m0).toBeCloseTo(mom.mean, 6);
    expect(Math.sqrt(m2 / m0 - (m1 / m0) ** 2) / (mom.sd / Math.sqrt(n))).toBeCloseTo(1, 4);
  });
  it('the grid size: a power of 2 from 256 to 65 536, spacing at most σ/(40√n)', () => {
    for (const [n, sd] of [[1, 1], [30, 0.3], [1000, 2], [1e4, 2], [1e4, 0.001]] as const) {
      const M = inversionGridSize(n, sd, 0, 10);
      expect(M & (M - 1)).toBe(0);
      expect(M).toBeGreaterThanOrEqual(256);
      expect(M).toBeLessThanOrEqual(65536);
      if (M < 65536) expect(10 / M).toBeLessThanOrEqual(sd / Math.sqrt(n) / 40);
    }
  });
  it('refuses a bad grid', () => {
    expect(() => meanDensityByInversion(uniformCf(0, 1), 0, 1, 2, 1000)).toThrow(RangeError);
    expect(() => meanDensityByInversion(uniformCf(0, 1), 1, 0, 2, 1024)).toThrow(RangeError);
  });
});

describe('distance to the normal: sup |F(μ + σz) − Φ(z)|', () => {
  // the exponential's mean is Gamma(n, rate n): shape n, scale 1/n, mean 1, sd 1/√n (H19.mjs §2)
  const expo = (n: number) => distanceToNormal((x) => gammaCdf(x, n, 1 / n), 1, 1 / Math.sqrt(n));
  it('n = 30: about 0.0243 (0.024291), the number in highlight 2', () => {
    const r = expo(30);
    expect(Math.abs(r.d - 0.02429142794175898)).toBeLessThan(5e-6);
    expect(r.z).toBeCloseTo(-0.015, 3);
  });
  it('n = 1, 4, 100: 0.158655, 0.066700, 0.013300', () => {
    expect(Math.abs(expo(1).d - 0.158655)).toBeLessThan(5e-6);
    expect(expo(1).z).toBeCloseTo(-1, 6); // where the exponential starts: Φ(−1)
    expect(Math.abs(expo(4).d - 0.0667)).toBeLessThan(5e-6);
    expect(Math.abs(expo(100).d - 0.0133)).toBeLessThan(5e-6);
  });
  it('is 0 for the normal itself', () => {
    expect(distanceToNormal((x) => normalCdf((x - 3) / 2), 3, 2).d).toBeLessThan(1e-15);
  });
});

// ---------------------------------------------------------------- pointwise bands (H12, design/pages/H12-law-of-large-numbers.md 数学)

/** P(X < a) for X ~ Binomial(m, P/100), exactly, as a pair of BigInts (numerator over 100^m). */
function exactBelow(m: number, P: number, a: number): { num: bigint; den: bigint } {
  const p = BigInt(P), q = BigInt(100 - P);
  let t = q ** BigInt(m), num = 0n; // t_k = C(m, k)·Pᵏ·Qᵐ⁻ᵏ
  for (let k = 0; k < a; k++) { num += t; t = (t * BigInt(m - k) * p) / (BigInt(k + 1) * q); }
  return { num, den: 100n ** BigInt(m) };
}
const ratio = (x: { num: bigint; den: bigint }) => Number((x.num * 10n ** 30n) / x.den) / 1e30;

describe('pointwise bands: the exact binomial band, n by n', () => {
  it('p = ½: [40, 60] at n = 100, [4902, 5098] at 10⁴, [49 690, 50 310] at 10⁵, with their coverage', () => {
    const b = binomialBand(100000, 0.5);
    const cover = (n: number) => 1 - b.below[n] - b.above[n];
    expect([b.lo[100], b.hi[100], b.lo[10000], b.hi[10000], b.lo[100000], b.hi[100000]]).toEqual([40, 60, 4902, 5098, 49690, 50310]);
    expect(cover(100).toFixed(4)).toBe('0.9648');
    expect(cover(10000).toFixed(4)).toBe('0.9512');
    expect(cover(100000).toFixed(4)).toBe('0.9504');
    expect([b.lo[0], b.hi[0], b.lo[1], b.hi[1], b.lo[10], b.hi[10]]).toEqual([0, 0, 0, 1, 2, 8]);
  });
  it('p = ½, n ≤ 600: every end equals the exact one (BigInt), and the tails are the exact tails', () => {
    const b = binomialBand(600, 0.5);
    for (let n = 1; n <= 600; n++) {
      // the largest a with 40·Σ_{k<a} C(n, k) ≤ 2ⁿ
      let a = 0, s = 0n, c = 1n;
      const tot = 1n << BigInt(n);
      while (40n * (s + c) <= tot) { s += c; a++; c = (c * BigInt(n - a + 1)) / BigInt(a); }
      expect(b.lo[n]).toBe(a);
      expect(b.hi[n]).toBe(n - a);
      expect(Math.abs(b.below[n] - Number((s * 10n ** 30n) / tot) / 1e30)).toBeLessThan(1e-15);
    }
  });
  it('agrees with exact rational tails at other p and larger n (error ~1e-16, far below the margin)', () => {
    for (const [P, m] of [[1, 3000], [37, 4000], [64, 2500], [99, 1500]] as const) {
      const b = binomialBand(m, P / 100);
      const below = ratio(exactBelow(m, P, b.lo[m]));
      expect(Math.abs(b.below[m] - below)).toBeLessThan(1e-14);
      expect(below).toBeLessThanOrEqual(0.025);
      expect(ratio(exactBelow(m, P, b.lo[m] + 1))).toBeGreaterThan(0.025);
    }
  });
  it('certified on the page’s grid p = 0.01 … 0.99: every decision is > 1e-11 from 0.025, every coverage ≥ 0.95', () => {
    let worst = Infinity;
    for (let k = 1; k <= 99; k++) {
      const b = binomialBand(100000, k / 100);
      worst = Math.min(worst, b.margin);
      for (let n = 0; n <= 100000; n += 997) {
        expect(b.below[n]).toBeLessThanOrEqual(0.025);
        expect(b.above[n]).toBeLessThanOrEqual(0.025);
        expect(b.lo[n]).toBeLessThanOrEqual(b.hi[n]);
      }
    }
    expect(worst).toBeGreaterThan(1e-11); // 3.8e-11 at p = 0.64, n = 35 641
  });
  it('the double walk equals a double-double walk (~32 digits) at every n, for p = ½ and the grid’s two hardest points', () => {
    // the whole grid is certified by scripts/certify-binomial-band.mjs (99 p, both ends, smallest margin 3.78e-11)
    for (const P of [36, 50, 64]) {
      const b = binomialBand(100000, P / 100), lo = ddLowerWalk(100000, P), up = ddLowerWalk(100000, 100 - P);
      expect(b.lo).toEqual(lo.lo);
      expect(b.hi).toEqual(Int32Array.from(up.lo, (a, n) => n - a));
      expect(Math.min(lo.margin, up.margin)).toBeGreaterThan(3e-11);
    }
  });
  it('symmetry: the band for 1 − p is the band for p reflected, n − X', () => {
    const a = binomialBand(5000, 0.3), b = binomialBand(5000, 0.7);
    for (let n = 0; n <= 5000; n += 7) {
      expect(a.lo[n]).toBe(n - b.hi[n]);
      expect(Math.abs(a.below[n] - b.above[n])).toBeLessThan(1e-13); // 1 − 0.7 is not exactly 0.3
    }
  });
  it('an exact tie (a tail equal to alpha/2) is not resolved: the margin says so', () => {
    // Binomial(1, 0.05), alpha 0.1: P(X > 0) = 0.05 = alpha/2 exactly; rounding decides, and the margin is ~0
    expect(binomialBand(1, 0.05, 0.1).margin).toBeLessThan(1e-15);
    expect(binomialBand(0, 0.5).lo[0]).toBe(0);
    expect(binomialBand(0, 0.5).hi[0]).toBe(0);
  });
  it('refuses p = 0, p = 1 and bad sizes', () => {
    expect(() => binomialBand(10, 0)).toThrow(RangeError);
    expect(() => binomialBand(10, 1)).toThrow(RangeError);
    expect(() => binomialBand(-1, 0.5)).toThrow(RangeError);
  });
});

describe('pointwise bands: exact quantiles of a lattice sum (the die)', () => {
  it('n ≤ 40 dice: equals the exact count (BigInt polynomial powers)', () => {
    const q = latticeSumLowerQuantiles(DIE, 40, 0.025);
    let ways: bigint[] = [1n]; // ways[t] = number of ways to make steps total t
    for (let n = 1; n <= 40; n++) {
      const next: bigint[] = Array(ways.length + 5).fill(0n);
      ways.forEach((w, t) => { for (let k = 0; k < 6; k++) next[t + k] += w; });
      ways = next;
      const tot = 6n ** BigInt(n);
      let a = 0, s = 0n;
      while (40n * (s + ways[a]) <= tot) { s += ways[a]; a++; }
      expect(q.lo[n]).toBe(a);
    }
  });
  it('n ≤ 1000: equals the quantiles of the full law (latticeSumPmf), for the die and a lopsided pmf', () => {
    for (const base of [DIE, [0.2, 0.5, 0.3]]) {
      const q = latticeSumLowerQuantiles(base, 1000, 0.025);
      for (const n of [1, 2, 3, 17, 100, 333, 1000]) {
        const p = latticeSumPmf(base, n);
        let F = 0, a = 0;
        while (F + p[a] <= 0.025) F += p[a++];
        expect(q.lo[n]).toBe(a);
        expect(Math.abs(q.below[n] - F)).toBeLessThan(1e-13);
      }
    }
  });
  it('the die at n = 100: totals 317 … 383 (lower end in steps + n)', () => {
    const q = latticeSumLowerQuantiles(DIE, 100, 0.025);
    expect(q.lo[100] + 100).toBe(317);
    expect(q.margin).toBeGreaterThan(1e-9);
  });
});
