import { describe, expect, it } from 'vitest';
import {
  c4, expoMedianMoments, expoOrderMoments, laplaceSumLaw, laplaceSumWeights, lawMoments, lawPasses, medianLaw, medianLawCheck, normalSdLaw, normalSdMoments,
  normalVarianceLaw, orderStatLaw, quantileOf, uniformMaxLaw, uniformMaxMoments, uniformMedianMoments, uniformOrderMoments, varOfS2,
  type PopulationLaw,
} from '../src/lib/math/sampling';
import { inversionGridSize, inversionLaw, irwinHallCdf, meanDensityByInversion, uniformCf } from '../src/lib/math/sums';
import { normalCdf, normalPdf, normalSf } from '../src/lib/math/special';
import { adaptiveSimpson } from '../src/lib/math/quad';
import { mulberry32, normal } from '../src/lib/math/rng';

// The sampling distributions behind H18 (design/pages/H18-sampling-distribution.md 数学). Values marked "H18.mjs" are
// the ones review/artifacts/page-directions/H18.mjs prints and checks.

const integrate = (f: (x: number) => number, a: number, b: number) => adaptiveSimpson(f, a, b, { tol: 1e-12, minDepth: 10 }).value;
const NORMAL: PopulationLaw = {
  F: (x) => normalCdf((x - 50) / 10), S: (x) => normalSf((x - 50) / 10), f: (x) => normalPdf((x - 50) / 10) / 10,
  support: [-Infinity, Infinity], range: [-20, 120],
};
const B = 10 / Math.SQRT2;
const LAPLACE: PopulationLaw = {
  F: (x) => (x < 50 ? 0.5 * Math.exp((x - 50) / B) : 1 - 0.5 * Math.exp(-(x - 50) / B)),
  S: (x) => (x < 50 ? 1 - 0.5 * Math.exp((x - 50) / B) : 0.5 * Math.exp(-(x - 50) / B)),
  f: (x) => Math.exp(-Math.abs(x - 50) / B) / (2 * B), support: [-Infinity, Infinity], range: [-150, 250], kinks: [50],
};
const UNIFORM: PopulationLaw = { F: (x) => Math.min(1, Math.max(0, x / 100)), f: (x) => (x >= 0 && x <= 100 ? 0.01 : 0), support: [0, 100], range: [0, 100] };
const EXPO: PopulationLaw = {
  F: (x) => (x <= 40 ? 0 : -Math.expm1(-(x - 40) / 10)), S: (x) => (x <= 40 ? 1 : Math.exp(-(x - 40) / 10)),
  f: (x) => (x < 40 ? 0 : Math.exp(-(x - 40) / 10) / 10), support: [40, Infinity], range: [40, 400],
};

describe('the sample variance and SD of a normal population', () => {
  it('c₄ (H18.mjs §2, §3)', () => {
    expect(c4(4)).toBeCloseTo(0.9213177319, 9);
    expect(c4(5)).toBeCloseTo(0.9399856030, 9);
    expect(c4(100)).toBeCloseTo(0.9974779761, 9);
    expect(c4(2)).toBeCloseTo(Math.sqrt(2 / Math.PI), 14);
  });
  it('s² ÷ (n − 1): density integrates to 1, mean σ² (H18.mjs §7); ÷ n: mean (n − 1)σ²/n = 80, variance by Var(s²)', () => {
    const L = normalVarianceLaw(10, 5, 4);
    expect(integrate(L.pdf, 0, 3000)).toBeCloseTo(1, 9);
    expect(integrate((v) => v * L.pdf(v), 0, 3000)).toBeCloseTo(100, 6);
    const Ln = normalVarianceLaw(10, 5, 5);
    const m = integrate((v) => v * Ln.pdf(v), 0, 3000);
    expect(m).toBeCloseTo(80, 6);
    expect(integrate((v) => (v - m) ** 2 * Ln.pdf(v), 0, 3000)).toBeCloseTo(varOfS2(10, 0, 5) * (4 / 5) ** 2, 4);
    expect(L.cdf(1e9)).toBe(1);
    // n = 2: one degree of freedom, the density of s² is unbounded at 0 and that of s is a half-normal's, finite
    expect(normalVarianceLaw(10, 2, 1).pdf(0)).toBe(Infinity);
    const s2 = normalSdLaw(10, 2, 1);
    expect(s2.pdf(0)).toBeCloseTo(s2.pdf(1e-9), 9);
    expect(normalSdLaw(10, 3, 2).pdf(0)).toBe(0);
  });
  it('s: density integrates to 1, mean c₄σ, SD σ√(1 − c₄²); ÷ n scales by √((n − 1)/n)', () => {
    for (const [n, d] of [[5, 4], [5, 5], [30, 29]] as const) {
      const L = normalSdLaw(10, n, d), mo = normalSdMoments(10, n, d);
      const q = lawMoments(L, 0, 80);
      expect(q.mass).toBeCloseTo(1, 9);
      expect(q.mean).toBeCloseTo(mo.mean, 7);
      expect(q.sd).toBeCloseTo(mo.sd, 7);
    }
    expect(normalSdMoments(10, 5, 4).mean).toBeCloseTo(9.399856, 5);
    expect(normalSdMoments(10, 5, 4).sd).toBeCloseTo(10 * Math.sqrt(1 - c4(5) ** 2), 12);
  });
});

describe('Var(s²) = σ⁴(2/(n − 1) + κ/n) for any population with a fourth moment', () => {
  it('equals an exact enumeration for a two-point population (p = 0.3, n = 4 and 6)', () => {
    const p = 0.3, q = 1 - p, sigma = Math.sqrt(p * q), kurt = (1 - 6 * p * q) / (p * q);
    for (const n of [4, 6]) {
      let m1 = 0, m2 = 0;
      for (let mask = 0; mask < 1 << n; mask++) {
        let ones = 0;
        for (let j = 0; j < n; j++) if (mask & (1 << j)) ones++;
        const pr = p ** ones * q ** (n - ones), xbar = ones / n;
        const s2 = (ones * (1 - xbar) ** 2 + (n - ones) * xbar ** 2) / (n - 1);
        m1 += pr * s2;
        m2 += pr * s2 * s2;
      }
      expect(m1).toBeCloseTo(sigma ** 2, 14);
      expect(m2 - m1 * m1).toBeCloseTo(varOfS2(sigma, kurt, n), 14);
    }
  });
  it('a normal population: 2σ⁴/(n − 1)', () => {
    expect(varOfS2(10, 0, 5)).toBe(5000);
  });
});

describe('order statistics and the median', () => {
  it('the uniform: the order statistic law is a scaled Beta; its moments as uniformOrderMoments', () => {
    for (const [n, k] of [[5, 3], [10, 1], [10, 10], [25, 13]] as const) {
      const L = orderStatLaw(UNIFORM.F, UNIFORM.f, n, k, [0, 100]);
      const q = lawMoments(L, 0, 100), o = uniformOrderMoments(n, k, 0, 100);
      expect(q.mass).toBeCloseTo(1, 9);
      expect(q.mean).toBeCloseTo(o.mean, 7);
      expect(q.sd).toBeCloseTo(Math.sqrt(o.variance), 7);
    }
  });
  it('the exponential: means and variances by Rényi’s representation', () => {
    for (const [n, k] of [[5, 3], [25, 13], [10, 10]] as const) {
      const L = orderStatLaw(EXPO.F, EXPO.f, n, k, [40, Infinity]);
      const q = lawMoments(L, 40, 400), o = expoOrderMoments(n, k, 10, 40);
      expect(q.mean).toBeCloseTo(o.mean, 7);
      expect(q.sd).toBeCloseTo(Math.sqrt(o.variance), 6);
    }
    // the median of 25 is biased: E = 40 + 10(H₂₅ − H₁₂) ≈ 47.127 against the population's 40 + 10 ln 2 ≈ 46.931
    const e = expoMedianMoments(25, 10, 40);
    expect(e.mean).toBeCloseTo(47.1275, 4);
    expect(e.mean - (40 + 10 * Math.LN2)).toBeCloseTo(0.196, 3);
  });
  it('the maximum of an Exp(1): variance Σ 1/k², 1.4636 at n = 5 and 1.6350 at n = 100 — it grows (H18.mjs §6)', () => {
    expect(expoOrderMoments(5, 5, 1).variance).toBeCloseTo(1.4636111, 7);
    expect(expoOrderMoments(100, 100, 1).variance).toBeCloseTo(1.6349839, 7);
  });
  it('the median of 25: standard error 2.485 (normal, σ = 10) and 1.638 (Laplace, σ = 10) (H18.mjs §5)', () => {
    const n = lawMoments(medianLaw(NORMAL, 25), -20, 120);
    expect(n.mass).toBeCloseTo(1, 9);
    expect(n.mean).toBeCloseTo(50, 9);
    expect(n.sd).toBeCloseTo(2.4848794142, 7);
    const l = lawMoments(medianLaw(LAPLACE, 25), -150, 250, [50]);
    expect(l.sd).toBeCloseTo(1.6382582532, 7);
    expect(n.sd / 2).toBeCloseTo(1.242, 3);
    expect(l.sd / 2).toBeCloseTo(0.819, 3);
  });
  it('even n: the midpoint law integrates to 1 (1e-6), is symmetric for a symmetric population, matches the closed-form moments', () => {
    for (const n of [2, 10, 26]) {
      const L = medianLaw(UNIFORM, n);
      expect(L.accuracy).toBe('numeric');
      expect(Math.abs(L.mass! - 1)).toBeLessThan(1e-6);
      const q = L.moments!, u = uniformMedianMoments(n, 0, 100);
      expect(q.mean).toBeCloseTo(50, 6);
      expect(q.sd / u.sd).toBeCloseTo(1, 5);
    }
    for (const n of [4, 10, 50]) {
      const L = medianLaw(EXPO, n);
      expect(Math.abs(L.mass! - 1)).toBeLessThan(1e-6);
      const q = L.moments!, e = expoMedianMoments(n, 10, 40);
      expect(q.mean).toBeCloseTo(e.mean, 4);
      expect(q.sd / e.sd).toBeCloseTo(1, 5);
    }
    const L = medianLaw(NORMAL, 200);
    expect(Math.abs(L.mass! - 1)).toBeLessThan(1e-6);
    for (const d of [0.3, 1, 2]) expect(L.pdf(50 - d)).toBeCloseTo(L.pdf(50 + d), 9);
    const lap = medianLaw(LAPLACE, 10);
    expect(Math.abs(lap.mass! - 1)).toBeLessThan(1e-6);
  });
  it('even n, normal: the numeric law agrees with a seeded simulation of 20 000 medians of 10 (SD within 2%)', () => {
    const rand = mulberry32(11), meds: number[] = [];
    for (let i = 0; i < 20000; i++) {
      const s = Array.from({ length: 10 }, () => normal(rand, 50, 10)).sort((a, b) => a - b);
      meds.push((s[4] + s[5]) / 2);
    }
    const m = meds.reduce((a, b) => a + b, 0) / meds.length;
    const sd = Math.sqrt(meds.reduce((a, b) => a + (b - m) ** 2, 0) / (meds.length - 1));
    const q = medianLaw(NORMAL, 10).moments!;
    expect(Math.abs(sd / q.sd - 1)).toBeLessThan(0.02);
  });
  it('n = 2: the midpoint is the mean of two — N(50, 50) exactly for the normal (Codex design review: a benchmark)', () => {
    const L = medianLaw(NORMAL, 2);
    for (const x of [35, 45, 50, 57, 66]) {
      // the CDF is the trapezoid of the grid's polyline: O(dx²), about 1e-5 here; the moments are Simpson's
      expect(L.cdf(x)).toBeCloseTo(normalCdf((x - 50) / Math.sqrt(50)), 4);
      expect(L.pdf(x) / (normalPdf((x - 50) / Math.sqrt(50)) / Math.sqrt(50))).toBeCloseTo(1, 3);
    }
    expect(L.moments!.sd).toBeCloseTo(Math.sqrt(50), 6);
  });
  it('the refinement check: a grid twice as fine agrees (CDF 5e-5, moments 1e-5: the page’s rule is 1e-4) for the four populations at n = 10 and 200; lawPasses', () => {
    for (const P of [NORMAL, UNIFORM, EXPO, LAPLACE]) for (const n of [10, 200]) {
      const L = medianLaw(P, n), c = medianLawCheck(P, n, L);
      expect(c.cdf).toBeLessThan(5e-5);
      expect(c.sd).toBeLessThan(1e-5);
      expect(lawPasses(L, c)).toBe(true);
    }
    expect(lawPasses({ ...medianLaw(NORMAL, 4), mass: 0.99 })).toBe(false);
    expect(lawPasses({ ...medianLaw(NORMAL, 4), mass: NaN })).toBe(false);
  });
  it('the uniform median moments: odd n by Beta, even n by the covariance of neighbours', () => {
    expect(uniformMedianMoments(5, 0, 100).sd).toBeCloseTo(100 * Math.sqrt((3 * 3) / (36 * 7)), 12);
    // n = 2: the midpoint of two uniforms is triangular on [0, 100], SD 100/√24
    expect(uniformMedianMoments(2, 0, 100).sd).toBeCloseTo(100 / Math.sqrt(24), 12);
  });
  it('quantileOf: the exponential’s median 40 + 10 ln 2', () => {
    expect(quantileOf(EXPO.F, 0.5, 40, 400)).toBeCloseTo(40 + 10 * Math.LN2, 9);
  });
});

describe('the maximum of a uniform', () => {
  it('E = 100n/(n + 1), variance 198.4 at n = 5 and 0.961 at n = 100 (H18.mjs §4, §7)', () => {
    const L = uniformMaxLaw(5, 0, 100);
    expect(integrate(L.pdf, 0, 100)).toBeCloseTo(1, 12);
    expect(integrate((m) => m * L.pdf(m), 0, 100)).toBeCloseTo(500 / 6, 9);
    expect(uniformMaxMoments(5, 0, 100).mean).toBeCloseTo(83.3333333, 6);
    expect(uniformMaxMoments(5, 0, 100).sd ** 2).toBeCloseTo(198.4127, 3);
    expect(uniformMaxMoments(100, 0, 100).sd ** 2).toBeCloseTo(0.9610746, 6);
  });
});

describe('the sum of n Laplace draws', () => {
  it('weights add up to ½; n = 1 is the Laplace density; n = 2 is (1 + |s|)e^{−|s|}/4', () => {
    for (const n of [1, 2, 7, 50, 200]) expect([...laplaceSumWeights(n)].reduce((a, b) => a + b, 0)).toBeCloseTo(0.5, 13);
    const one = laplaceSumLaw(1, 2);
    for (const s of [-3, -0.5, 0, 1, 4]) expect(one.pdf(s)).toBeCloseTo(Math.exp(-Math.abs(s) / 2) / 4, 14);
    const two = laplaceSumLaw(2, 1);
    for (const s of [-3, -0.5, 0, 1, 4]) expect(two.pdf(s)).toBeCloseTo(((1 + Math.abs(s)) * Math.exp(-Math.abs(s))) / 4, 14);
    expect(two.cdf(0)).toBe(0.5);
  });
  it('integrates to 1 with variance 2nb², up to n = 200; the CDF is the integral of the density', () => {
    for (const n of [3, 25, 200]) {
      const L = laplaceSumLaw(n, B), r = 12 * B * Math.sqrt(n) + 40 * B;
      const q = lawMoments(L, -r, r, [0]);
      expect(q.mass).toBeCloseTo(1, 9);
      expect(q.sd ** 2 / (2 * n * B * B)).toBeCloseTo(1, 8);
      expect(L.cdf(3) - L.cdf(-1)).toBeCloseTo(integrate(L.pdf, -1, 0) + integrate(L.pdf, 0, 3), 10);
    }
  });
  it('agrees with a numeric convolution of three Laplace densities', () => {
    const f1 = (x: number) => Math.exp(-Math.abs(x)) / 2;
    const two = laplaceSumLaw(2, 1).pdf;
    const conv = (s: number) => integrate((x) => f1(x) * two(s - x), -40, 0) + integrate((x) => f1(x) * two(s - x), 0, 40);
    const three = laplaceSumLaw(3, 1);
    for (const s of [0, 0.7, 2.5, -4]) expect(three.pdf(s)).toBeCloseTo(conv(s), 9);
  });
});

describe('the inversion law (numeric)', () => {
  it('agrees with Irwin–Hall at n = 50 to 5e-5 (the grid’s trapezoid CDF)', () => {
    const n = 50, M = inversionGridSize(n, 1 / Math.sqrt(12), 0, 1);
    const L = inversionLaw(meanDensityByInversion(uniformCf(0, 1), 0, 1, n, M), [0, 1]);
    expect(L.accuracy).toBe('numeric');
    for (const x of [0.4, 0.45, 0.5, 0.53, 0.6]) expect(L.cdf(x)).toBeCloseTo(irwinHallCdf(n * x, n), 4);
    expect(L.cdf(1)).toBe(1);
  });
});
