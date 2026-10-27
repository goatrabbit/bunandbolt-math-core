// The one random-number generator of the statistics pages, and its samplers. This file is the specification the
// pages and the direction-draft checks share (review/artifacts/page-directions/H19-sim-core.mjs, group8-stat-lib.mjs):
// a seeded simulation quoted in a highlight is only true for this generator, these samplers and this order of draws,
// so none of them may change without bumping ALGORITHM_VERSION (it goes into every share link with the seed) and
// re-running every check in that folder.
//
// - mulberry32: 32-bit seed, period 2³², uniform on [0, 1) with 2⁻³² resolution. Fast and fully reproducible across
//   browsers (only 32-bit integer operations); not for cryptography.
// - Box–Muller, cos branch only: one normal per two uniforms; u is redrawn while it is exactly 0.

export const ALGORITHM_VERSION = 'stats-rng-v1';

export type Rand = () => number;

/** mulberry32: a uniform generator on [0, 1). */
export function mulberry32(seed: number): Rand {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- samplers: each consumes draws from `rand` in exactly the order written ----

/** Standard normal by Box–Muller (cos branch only; redraw u while it is 0). Two uniforms per call (or more on a 0). */
export function gauss(rand: Rand): number {
  let u = 0;
  while (u === 0) u = rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const normal = (rand: Rand, mu: number, sigma: number) => mu + sigma * gauss(rand);
/** 1 with probability p (one uniform: u < p). */
export const bernoulli = (rand: Rand, p: number) => (rand() < p ? 1 : 0);
/** A fair die, 1 … 6 (one uniform). */
export const die = (rand: Rand) => 1 + Math.floor(6 * rand());
/** Exp(1) as −ln(1 − u): 1 − u is in (0, 1], so the logarithm is finite. */
export const expo = (rand: Rand) => -Math.log(1 - rand());
/** Laplace(0, b) as b·(E₁ − E₂), two Exp(1) draws in that order. */
export const laplace = (rand: Rand, b: number) => b * (expo(rand) - expo(rand));
/** Standard Cauchy as tan(π(u − ½)). */
export const cauchy = (rand: Rand) => Math.tan(Math.PI * (rand() - 0.5));
export const uniform = (rand: Rand, a: number, b: number) => a + (b - a) * rand();
/** Equal mixture of N(m1, s²) and N(m2, s²): one uniform picks the hump (u < ½ → first), then Box–Muller. */
export const bimodal = (rand: Rand, m1: number, m2: number, s: number) => (rand() < 0.5 ? normal(rand, m1, s) : normal(rand, m2, s));
/**
 * A finite discrete law (added 2026-10-07 for H19; the samplers above are unchanged, so the version is too): one
 * uniform u, and the value is `values[i]` for the first i with u < `cum[i]` (the cumulative probabilities, in the
 * order given, the last one 1). The order of `values` is part of the specification: H19's symmetric rare event is
 * `[−10, 10, 0]` with `[0.005, 0.01, 1]` (u < 0.005 → −10, u < 0.01 → +10, otherwise 0).
 */
export function discrete(rand: Rand, values: readonly number[], cum: readonly number[]): number {
  const u = rand();
  for (let i = 0; i < cum.length - 1; i++) if (u < cum[i]) return values[i];
  return values[values.length - 1];
}

/**
 * A population the reader draws: density heights h[i] at x[i], piecewise linear between grid points. One uniform per
 * draw, inverted through the exact piecewise-quadratic CDF of that polyline. Any scale of heights works (they are
 * rescaled by a power of two).
 * - Thrown (a RangeError) means a bug in the page, whose grid this is: fewer than two points, a height count that does
 *   not match, x not finite or not strictly increasing with every gap at least 1e-300, a height that is not finite.
 * - Returned null means the reader drew something with no density — every height 0, or any height below 0 (the page
 *   clips at the axis first if it wants that) — and the page shows that instead of sampling. It never throws for that.
 */
export function drawnSampler(x: number[], h: number[]): ((rand: Rand) => number) | null {
  if (x.length < 2 || h.length !== x.length) throw new RangeError('drawnSampler: need two or more points, one height each');
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(h[i])) throw new RangeError('drawnSampler: positions and heights must be finite');
    if (i > 0 && !(x[i] - x[i - 1] >= 1e-300)) throw new RangeError('drawnSampler: positions must increase by at least 1e-300');
  }
  if (h.some((v) => v < 0) || !h.some((v) => v > 0)) return null; // what the reader drew has no density
  // heights rescaled by a power of two (exact, so every draw is unchanged) to keep h² clear of under- and overflow
  const top = Math.max(...h.map(Math.abs));
  const k = top > 0 && Number.isFinite(top) ? Math.round(Math.log2(top)) : 0;
  h = h.map((v) => v * 2 ** Math.trunc(-k / 2) * 2 ** (-k - Math.trunc(-k / 2)));
  const seg: number[] = [];
  let total = 0;
  for (let i = 0; i + 1 < x.length; i++) {
    seg.push(total);
    total += ((h[i] + h[i + 1]) / 2) * (x[i + 1] - x[i]);
  }
  return (rand) => {
    const target = rand() * total;
    let i = seg.length - 1;
    while (i > 0 && seg[i] > target) i--;
    const w = x[i + 1] - x[i], h0 = h[i], h1 = h[i + 1], r = target - seg[i], slope = (h1 - h0) / w;
    // the root of h0·t + slope·t²/2 = r in the form 2r/(h0 + √(h0² + 2·slope·r)): no cancellation and no division by
    // the slope, so it holds at any density scale (an absolute flat-slope cutoff misread a density of height 1e-13)
    const den = h0 + Math.sqrt(Math.max(0, h0 * h0 + 2 * slope * r));
    const t = den > 0 ? (2 * r) / den : 0;
    return x[i] + Math.min(w, Math.max(0, t));
  };
}

// ---- small statistics on arrays (used with the samplers) ----

export function mean(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

/** Standard deviation with `ddof` (1: sample s, 0: population σ). Two-pass, so large offsets do not cancel. */
export function sd(a: ArrayLike<number>, ddof = 1): number {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / (a.length - ddof));
}

/** Moment skewness g₁ = m₃ / m₂^{3/2} (population moments). */
export function skewness(a: ArrayLike<number>): number {
  const m = mean(a);
  let s2 = 0, s3 = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    s2 += d * d;
    s3 += d * d * d;
  }
  s2 /= a.length;
  s3 /= a.length;
  return s3 / s2 ** 1.5;
}

export function median(a: ArrayLike<number>): number {
  const s = Array.from(a).sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
