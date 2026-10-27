// Distributions of the sum and the mean of n independent, identically distributed draws — what the central limit
// theorem page (H19) draws as the exact distribution of the sample mean, and the numbers its readouts and highlights
// quote. Each population gets the most exact method it has, and the page labels the result accordingly
// (design/pages/H19-central-limit-theorem.md 数学):
//
// - Lattice populations (a die, a rare event): `latticeSumPmf` adds one copy at a time; every probability is a plain
//   sum of non-negative products (no subtraction, no transform), so the only error is rounding — exact in the page's
//   four-way classification. `binomialPmf` and `symmetricTrinomialPmf` are closed forms for the two rare events.
// - The uniform: `irwinHallPdf` / `irwinHallCdf`, the alternating closed form, trusted up to n = 50 (there the largest
//   term is about 1e7 and about nine digits survive double precision; at n = 80 it is 1e12 and the form is useless).
// - Two normal humps: the mean is a finite normal mixture (`meanOfTwoNormals`), exact.
// - A drawn (piecewise-linear) density, and the uniform past n = 50: numeric — `meanDensityByInversion` evaluates
//   φ(t/n)ⁿ on a grid of frequencies and inverts it with an FFT. The population lives in [a, b], so its mean does too,
//   and with the period L = b − a nothing aliases; the error is the truncation to M frequencies. `inversionCheck`
//   is the page's rule for saying "approximate": repeat with the period doubled and the spacing halved, compare CDFs.
// - `distanceToNormal`: sup_z |F(μ + σz) − Φ(z)| for a continuous F (the readout "distance to normal").
import { fft, nextPow2 } from './fft';
import { lgamma, normalCdf, normalPdf } from './special';

// ---------------------------------------------------------------- lattice sums

/** Below this a probability is set to 0 while summing: it keeps the recursion out of subnormal numbers (which are
 *  slow on some machines) and changes no probability by more than 1e-300. */
const TINY = 1e-300;

/**
 * The pmf of the sum of n iid copies of a lattice variable whose pmf is base[k] on k = 0 … m − 1 (lattice steps); the
 * result has length (m − 1)·n + 1 (index = the sum in steps). One copy is added at a time: each new probability is a
 * plain sum of non-negative products, so nothing cancels. Only the non-zero stretch is visited (far tails underflow
 * to 0, which for the die at n = 10 000 is three quarters of the support). n = 0 gives [1].
 */
export function latticeSumPmf(base: ArrayLike<number>, n: number): Float64Array {
  const m = base.length;
  if (m < 1 || !Number.isInteger(n) || n < 0) throw new RangeError('latticeSumPmf: need a non-empty pmf and a whole n ≥ 0');
  const size = (m - 1) * n + 1;
  let cur = new Float64Array(size), next = new Float64Array(size);
  cur[0] = 1;
  let lo = 0, hi = 0; // the non-zero stretch of cur
  const js: number[] = [], bs: number[] = [];
  for (let j = 0; j < m; j++) if (base[j] > 0) { js.push(j); bs.push(base[j]); }
  for (let step = 0; step < n; step++) {
    const nlo = lo + js[0], nhi = hi + js[js.length - 1];
    next.fill(0, nlo, nhi + 1);
    for (let t = 0; t < js.length; t++) {
      const j = js[t], b = bs[t];
      for (let i = lo; i <= hi; i++) next[i + j] += cur[i] * b;
    }
    // trim what has underflowed at the two ends (exact zeros after the flush)
    lo = nlo;
    hi = nhi;
    while (lo < hi && next[lo] < TINY) next[lo++] = 0;
    while (hi > lo && next[hi] < TINY) next[hi--] = 0;
    const t = cur; cur = next; next = t;
  }
  // only [lo, hi] was kept current; the buffers swap, so older values may linger outside it
  cur.fill(0, 0, lo);
  cur.fill(0, hi + 1);
  return cur;
}

/** δ(z) = ln Γ(z) − [(z − ½) ln z − z + ½ ln 2π], the remainder of Stirling's formula (≈ 1/(12z)): the series for
 *  z ≥ 10 (error < 1e-17), lgamma below (where nothing large cancels). */
function stirlingDelta(z: number): number {
  if (z < 10) return lgamma(z) - ((z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI));
  const z2 = z * z;
  return 1 / (12 * z) - 1 / (360 * z * z2) + 1 / (1260 * z * z2 * z2) - 1 / (1680 * z * z2 ** 3) + 1 / (1188 * z * z2 ** 4)
    - 691 / (360360 * z * z2 ** 5) + 1 / (156 * z * z2 ** 6);
}

/** x·ln(x/m) + m − x without cancellation when x is near m (Loader's bd0: a series in v = (x − m)/(x + m)). */
function bd0(x: number, m: number): number {
  if (Math.abs(x - m) < 0.1 * (x + m)) {
    const v = (x - m) / (x + m), v2 = v * v;
    let s = (x - m) * v, ej = 2 * x * v;
    for (let j = 1; j < 1000; j++) {
      ej *= v2;
      const s1 = s + ej / (2 * j + 1);
      if (s1 === s) return s1;
      s = s1;
    }
    return s;
  }
  return x * Math.log(x / m) + m - x;
}

/** ln P(K = k) for K ~ Binomial(n, p), 0 < k < n, 0 < p < 1, by Loader's saddle-point form (Loader 2000, the form R's
 *  dbinom uses): ½ ln(n/(2πk(n − k))) + δ(n) − δ(k) − δ(n − k) − bd0(k, np) − bd0(n − k, nq). Nothing large is
 *  subtracted, so it keeps full relative accuracy at any n (ln C(n, k) from three lgamma values loses about 1e-11 at
 *  n = 10 000: each is near 10⁵, where one ulp is 1.5e-11). */
function logBinom(n: number, k: number, p: number, q: number): number {
  return 0.5 * Math.log(n / (2 * Math.PI * k * (n - k))) + stirlingDelta(n) - stirlingDelta(k) - stirlingDelta(n - k) - bd0(k, n * p) - bd0(n - k, n * q);
}

/** Binomial(n, p) probabilities, k = 0 … n: the closed form C(n, k)pᵏ(1 − p)ⁿ⁻ᵏ, evaluated at the mode (saddle-point
 *  form, full relative accuracy) and outward by the exact ratio of neighbours; probabilities that underflow are 0. */
export function binomialPmf(n: number, p: number): Float64Array {
  if (!Number.isInteger(n) || n < 0 || !(p >= 0 && p <= 1)) throw new RangeError('binomialPmf: need a whole n ≥ 0 and 0 ≤ p ≤ 1');
  const out = new Float64Array(n + 1);
  if (p === 0) { out[0] = 1; return out; }
  if (p === 1) { out[n] = 1; return out; }
  const q = 1 - p;
  const mode = Math.min(n, Math.max(0, Math.floor((n + 1) * p)));
  out[mode] = mode === 0 ? Math.exp(n * Math.log1p(-p)) : mode === n ? Math.exp(n * Math.log(p)) : Math.exp(logBinom(n, mode, p, q));
  const up = p / q, down = q / p;
  for (let k = mode; k < n && out[k] > 0; k++) out[k + 1] = (out[k] * (n - k) * up) / (k + 1);
  for (let k = mode; k > 0 && out[k] > 0; k--) out[k - 1] = (out[k] * k * down) / (n - k + 1);
  return out;
}

/**
 * X ∈ {−1, 0, +1} with P(+1) = P(−1) = q: the pmf of D = X₁ + … + Xₙ on d = −n … n (index d + n). The number of
 * non-zero draws is m ~ Binomial(n, 2q) and, given m, the number of +1s is Binomial(m, ½), so
 * P(D = d) = Σₘ Bin(m; n, 2q)·Bin((m + d)/2; m, ½) over m ≡ d (mod 2), |d| ≤ m — every term a closed form. Only the m
 * where Bin(m; n, 2q) exceeds 1e-300 of its largest value are visited (a narrow band when q is small).
 */
export function symmetricTrinomialPmf(n: number, q: number): Float64Array {
  if (!Number.isInteger(n) || n < 0 || !(q >= 0 && q <= 0.5)) throw new RangeError('symmetricTrinomialPmf: need a whole n ≥ 0 and 0 ≤ q ≤ ½');
  const out = new Float64Array(2 * n + 1);
  const bm = binomialPmf(n, 2 * q);
  let top = 0;
  for (let m = 0; m <= n; m++) top = Math.max(top, bm[m]);
  for (let m = 0; m <= n; m++) {
    if (bm[m] <= top * 1e-300 || bm[m] === 0) continue;
    const half = binomialPmf(m, 0.5); // j plus signs: d = 2j − m
    for (let j = 0; j <= m; j++) out[2 * j - m + n] += bm[m] * half[j];
  }
  return out;
}

// ---------------------------------------------------------------- the uniform: Irwin–Hall

/** The alternating closed form keeps about nine digits at n = 50 and none worth having by n = 80. */
export const IRWIN_HALL_MAX = 50;

const checkIH = (n: number) => {
  if (!Number.isInteger(n) || n < 1 || n > IRWIN_HALL_MAX) throw new RangeError(`Irwin–Hall closed form: n must be a whole number from 1 to ${IRWIN_HALL_MAX}`);
};
const choose = (n: number, k: number) => { // exact for n ≤ 50 (C(50, 25) ≈ 1.3e14 < 2⁵³)
  let c = 1;
  for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i;
  return Math.round(c);
};
const factorial = (n: number) => { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; };

/** The density of the sum of n U(0, 1) at x: (1/(n − 1)!) Σ_{k ≤ ⌊x⌋} (−1)ᵏ C(n, k)(x − k)ⁿ⁻¹, through the symmetry
 *  x → n − x so ⌊x⌋ ≤ n/2; 0 outside [0, n]. n from 1 to IRWIN_HALL_MAX. */
export function irwinHallPdf(x: number, n: number): number {
  checkIH(n);
  if (!(x >= 0 && x <= n)) return 0;
  if (x > n / 2) x = n - x;
  if (n === 1) return 1;
  const f = factorial(n - 1);
  let s = 0;
  for (let k = 0; k <= Math.floor(x); k++) s += (k % 2 ? -1 : 1) * choose(n, k) * (x - k) ** (n - 1);
  return Math.max(0, s / f);
}

/** The CDF of the sum of n U(0, 1): (1/n!) Σ_{k ≤ ⌊x⌋} (−1)ᵏ C(n, k)(x − k)ⁿ, and 1 − F(n − x) above n/2. */
export function irwinHallCdf(x: number, n: number): number {
  checkIH(n);
  if (Number.isNaN(x)) return NaN;
  if (x <= 0) return 0;
  if (x >= n) return 1;
  const lower = (y: number) => {
    let s = 0;
    for (let k = 0; k <= Math.floor(y); k++) s += (k % 2 ? -1 : 1) * choose(n, k) * (y - k) ** n;
    return Math.min(1, Math.max(0, s / factorial(n)));
  };
  return x > n / 2 ? 1 - lower(n - x) : lower(x);
}

// ---------------------------------------------------------------- two normal humps

/** A normal mixture with one common standard deviation: weights w, means m. */
export interface Mixture { w: Float64Array; m: Float64Array; s: number }

/** The mean of n draws from ½N(m1, s²) + ½N(m2, s²): with k draws from the first hump (k ~ Binomial(n, ½)) it is
 *  N((k·m1 + (n − k)·m2)/n, s²/n), so the mean is that finite mixture — exact. Components below 1e-18 of the largest
 *  weight are left out (they are 0 on any picture and in any printed digit). */
export function meanOfTwoNormals(n: number, m1: number, m2: number, s: number): Mixture {
  if (!Number.isInteger(n) || n < 1 || !(s > 0)) throw new RangeError('meanOfTwoNormals: need a whole n ≥ 1 and s > 0');
  const b = binomialPmf(n, 0.5);
  let top = 0;
  for (let k = 0; k <= n; k++) top = Math.max(top, b[k]);
  const ks: number[] = [];
  for (let k = 0; k <= n; k++) if (b[k] >= top * 1e-18) ks.push(k);
  const w = new Float64Array(ks.length), m = new Float64Array(ks.length);
  ks.forEach((k, i) => { w[i] = b[k]; m[i] = (k * m1 + (n - k) * m2) / n; });
  return { w, m, s: s / Math.sqrt(n) };
}

export function mixturePdf(mx: Mixture, x: number): number {
  let f = 0;
  for (let i = 0; i < mx.w.length; i++) f += mx.w[i] * normalPdf((x - mx.m[i]) / mx.s);
  return f / mx.s;
}

export function mixtureCdf(mx: Mixture, x: number): number {
  let F = 0;
  for (let i = 0; i < mx.w.length; i++) F += mx.w[i] * normalCdf((x - mx.m[i]) / mx.s);
  return Math.min(1, F);
}

// ---------------------------------------------------------------- a drawn density (piecewise linear)

/** Three-point Gauss–Legendre on [0, 1]: exact for polynomials of degree ≤ 5 (a segment's x⁴ times a linear height). */
const GL = [[0.5 - Math.sqrt(15) / 10, 5 / 18], [0.5, 8 / 18], [0.5 + Math.sqrt(15) / 10, 5 / 18]] as const;

function checkPolyline(x: ArrayLike<number>, h: ArrayLike<number>) {
  if (x.length < 2 || h.length !== x.length) throw new RangeError('polyline: need two or more points, one height each');
  for (let i = 1; i < x.length; i++) if (!(x[i] > x[i - 1])) throw new RangeError('polyline: positions must increase');
}

/**
 * Exact moments of the density through (x[i], h[i]), linear between points, normalised to area 1: its area before
 * normalising, mean, standard deviation, skewness and excess kurtosis. Each segment's integrals of (x − c)ᵏ·height
 * (degree ≤ 5) use three-point Gauss–Legendre, which is exact for them and never subtracts large powers; the central
 * moments are taken about the mean found in a first pass.
 */
export function polylineMoments(x: ArrayLike<number>, h: ArrayLike<number>): { area: number; mean: number; sd: number; skew: number; kurt: number } {
  checkPolyline(x, h);
  const integrate = (c: number, kmax: number) => {
    const out = new Array(kmax + 1).fill(0);
    for (let i = 0; i + 1 < x.length; i++) {
      const w = x[i + 1] - x[i];
      for (const [v, wt] of GL) {
        const u = x[i] + w * v - c, g = h[i] + (h[i + 1] - h[i]) * v;
        let p = wt * w * g;
        for (let k = 0; k <= kmax; k++) { out[k] += p; p *= u; }
      }
    }
    return out;
  };
  const c0 = (x[0] + x[x.length - 1]) / 2;
  const [area, s1] = integrate(c0, 1);
  if (!(area > 0)) throw new RangeError('polylineMoments: the density has no area');
  const mean = c0 + s1 / area;
  const [, , m2, m3, m4] = integrate(mean, 4).map((v) => v / area);
  return { area, mean, sd: Math.sqrt(m2), skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) - 3 };
}

/** A characteristic function t ↦ E[e^{itX}] as [re, im]; `bound`, when given, is a decreasing upper bound of |φ(t)|
 *  for t > 0 (it lets the inversion stop once the remaining terms cannot matter). */
export interface CharFn {
  (t: number): [number, number];
  bound?: (t: number) => number;
}

/** The characteristic function of the piecewise-linear density through (x[i], h[i]) normalised to area 1, closed form
 *  per segment: w·e^{itx₀}·(h₀∫₀¹(1 − v)e^{iθv}dv + h₁∫₀¹ v e^{iθv}dv) with θ = t·w. Its `bound` is
 *  (|h₀| + |h_last|)/(A|t|) + (Σ |slope jumps|)/(A t²) (two integrations by parts). */
export function polylineCf(x: ArrayLike<number>, h: ArrayLike<number>): CharFn {
  checkPolyline(x, h);
  const N = x.length;
  let area = 0;
  for (let i = 0; i + 1 < N; i++) area += ((h[i] + h[i + 1]) / 2) * (x[i + 1] - x[i]);
  if (!(area > 0)) throw new RangeError('polylineCf: the density has no area');
  const X = Float64Array.from(x as ArrayLike<number>), H = Float64Array.from(h as ArrayLike<number>);
  // per segment θ = t·w: J₀ = ∫₀¹(1 − v)e^{iθv}dv, I₁ = ∫₀¹ v e^{iθv}dv — a series for |θ| < 1 (no cancellation),
  // the closed form above (I₀ = (e^{iθ} − 1)/(iθ), I₁ = e^{iθ}/(iθ) + (e^{iθ} − 1)/θ², J₀ = I₀ − I₁); inlined, no
  // allocation: the inversion calls this up to 2¹⁸ times per picture
  const cf: CharFn = (t: number) => {
    let re = 0, im = 0;
    for (let i = 0; i + 1 < N; i++) {
      const w = X[i + 1] - X[i], th = t * w;
      let j0r: number, j0i: number, i1r: number, i1i: number;
      if (Math.abs(th) < 1) {
        // I₀ = Σ (iθ)ᵐ/(m+1)!, I₁ = Σ (iθ)ᵐ/(m!(m+2))
        let pr = 1, pi = 0, i0r = 0, i0i = 0;
        i1r = 0; i1i = 0;
        for (let m = 0; m < 30; m++) {
          i0r += pr / (m + 1); i0i += pi / (m + 1);
          i1r += pr / (m + 2); i1i += pi / (m + 2);
          const nr = (-pi * th) / (m + 1);
          pi = (pr * th) / (m + 1);
          pr = nr;
          if (Math.abs(pr) + Math.abs(pi) < 1e-18) break;
        }
        j0r = i0r - i1r; j0i = i0i - i1i;
      } else {
        const c = Math.cos(th), s = Math.sin(th), t2 = th * th;
        i1r = s / th + (c - 1) / t2;
        i1i = -c / th + s / t2;
        j0r = s / th - i1r;
        j0i = (1 - c) / th - i1i;
      }
      const kr = H[i] * j0r + H[i + 1] * i1r, ki = H[i] * j0i + H[i + 1] * i1i;
      const c = Math.cos(t * X[i]), s = Math.sin(t * X[i]);
      re += w * (kr * c - ki * s);
      im += w * (kr * s + ki * c);
    }
    return [re / area, im / area];
  };
  let jumps = 0;
  const slope = (i: number) => (H[i + 1] - H[i]) / (X[i + 1] - X[i]);
  jumps += Math.abs(slope(0)) + Math.abs(slope(N - 2));
  for (let i = 1; i + 1 < N; i++) jumps += Math.abs(slope(i) - slope(i - 1));
  const ends = Math.abs(H[0]) + Math.abs(H[N - 1]);
  cf.bound = (t: number) => (t === 0 ? Infinity : ends / (area * Math.abs(t)) + jumps / (area * t * t));
  return cf;
}

/** The characteristic function of U(a, b): e^{it(a+b)/2}·sin(z)/z with z = t(b − a)/2; |φ| ≤ 2/(|t|(b − a)). */
export function uniformCf(a: number, b: number): CharFn {
  if (!(b > a)) throw new RangeError('uniformCf: need a < b');
  const cf: CharFn = (t: number) => {
    const z = (t * (b - a)) / 2, s = z === 0 ? 1 : Math.sin(z) / z, ph = (t * (a + b)) / 2;
    return [s * Math.cos(ph), s * Math.sin(ph)];
  };
  cf.bound = (t: number) => (t === 0 ? Infinity : 2 / (Math.abs(t) * (b - a)));
  return cf;
}

// ---------------------------------------------------------------- the mean by Fourier inversion (numeric)

/**
 * The density of the mean of n iid draws from a population supported in [a, b] with characteristic function `cf`.
 * The mean also lies in [a, b], so its density, periodised with period L = b − a, is itself on [a, a + L): no
 * aliasing. Its Fourier coefficients are φ_X̄(ωₖ)/L with ωₖ = 2πk/L and φ_X̄(ω) = φ(ω/n)ⁿ (the power taken through
 * modulus and argument, which is branch-free), k = −M/2 … M/2 − 1; one FFT gives the truncated series at
 * xⱼ = a + j·L/M, j = 0 … M − 1. The only error is that truncation (and rounding); negative values it leaves are set
 * to 0. When `cf.bound` is given, frequencies whose |φ_X̄| ≤ bound(ω/n)ⁿ < 1e-20 are skipped (they are below any
 * printed digit). M must be a power of 2. Numeric: the page draws it as a thin dashed line labelled "numeric".
 */
export function meanDensityByInversion(cf: CharFn, a: number, b: number, n: number, M: number): { x0: number; dx: number; f: Float64Array } {
  if (!(b > a) || !Number.isInteger(n) || n < 1) throw new RangeError('meanDensityByInversion: need a < b and a whole n ≥ 1');
  if (M < 2 || (M & (M - 1)) !== 0) throw new RangeError('meanDensityByInversion: M must be a power of 2');
  const L = b - a;
  const re = new Float64Array(M), im = new Float64Array(M);
  const put = (k: number) => {
    const w = (2 * Math.PI * k) / L;
    const [cr, ci] = cf(w / n);
    const r = Math.hypot(cr, ci) ** n, th = n * Math.atan2(ci, cr) - w * a; // φ_X̄(ω)·e^{−iωa}
    const j = k < 0 ? k + M : k;
    re[j] = r * Math.cos(th);
    im[j] = r * Math.sin(th);
  };
  put(0);
  for (let k = 1; k <= M / 2; k++) {
    const w = (2 * Math.PI * k) / L;
    if (cf.bound && cf.bound(w / n) ** n < 1e-20) break; // the bound decreases: every later term is smaller still
    if (k < M / 2) put(k);
    put(-k);
  }
  // fⱼ = (1/L) Σₖ cₖ e^{−2πi kj/M}: a forward transform
  fft(re, im, false);
  const f = new Float64Array(M);
  for (let j = 0; j < M; j++) f[j] = Math.max(0, re[j] / L);
  return { x0: a, dx: L / M, f };
}

/** The grid the page inverts on: spacing at most 1/40 of the mean's standard deviation σ/√n, at least 256 points, a
 *  power of 2, at most 2¹⁶. */
export function inversionGridSize(n: number, sd: number, a: number, b: number): number {
  const need = Math.ceil((40 * (b - a) * Math.sqrt(n)) / sd);
  return Math.min(65536, Math.max(256, nextPow2(need)));
}

/** Cumulative trapezoid of a grid density from its first point; F[j] at x0 + j·dx, with the period closing the last
 *  interval (f at b is f at a). */
function cdfOf(g: { f: Float64Array; dx: number }): Float64Array {
  const M = g.f.length, F = new Float64Array(M + 1);
  for (let j = 0; j < M; j++) F[j + 1] = F[j] + ((g.f[j] + g.f[(j + 1) % M]) / 2) * g.dx;
  return F;
}

/**
 * The page's rule for "approximate": invert again with the period doubled (on [a − L/2, b + L/2]) and the spacing
 * halved (4M points), and return the largest difference between the two CDFs at the coarse grid's points. Above 1e-4
 * the page adds "approximate" to the "numeric" label.
 */
export function inversionCheck(cf: CharFn, a: number, b: number, n: number, M: number): number {
  const L = b - a;
  const coarse = meanDensityByInversion(cf, a, b, n, M);
  const fine = meanDensityByInversion(cf, a - L / 2, b + L / 2, n, 4 * M);
  const C = cdfOf(coarse), F = cdfOf(fine);
  let d = 0;
  for (let j = 0; j <= M; j++) d = Math.max(d, Math.abs(C[j] - (F[M + 2 * j] - F[M])));
  return d;
}

/**
 * A law from an inversion grid (`meanDensityByInversion`): the density at x_j = x0 + j·dx (j < M) is periodic with
 * period M·dx, so the last cell runs from f[M−1] to f[0]; linear in between, the CDF its exact integral (the cumulative
 * trapezoid), both divided by the grid's total mass — the density drawn and the CDF the readouts use are one law with
 * area 1. Numeric. (H19's page-level `gridLaw`, 2026-10-08 into the library for H18; H19 keeps its own copy.)
 */
export function inversionLaw(g: { x0: number; dx: number; f: Float64Array }, support: [number, number]): { pdf: (x: number) => number; cdf: (x: number) => number; support: [number, number]; accuracy: 'numeric'; mass: number } {
  const { x0, dx, f } = g, M = f.length;
  const next = (j: number) => f[(j + 1) % M];
  const F = new Float64Array(M + 1);
  for (let j = 0; j < M; j++) F[j + 1] = F[j] + ((f[j] + next(j)) / 2) * dx;
  const total = F[M] > 0 ? F[M] : 1;
  const at = (x: number) => (x - x0) / dx;
  return {
    accuracy: 'numeric', support, mass: F[M],
    pdf: (x) => {
      if (!(x >= support[0] && x <= support[1])) return 0;
      const t = at(x), j = Math.min(M - 1, Math.max(0, Math.floor(t)));
      return (f[j] + (next(j) - f[j]) * (t - j)) / total;
    },
    cdf: (x) => {
      if (x <= support[0]) return 0;
      if (x >= support[1]) return 1;
      const t = at(x), j = Math.max(0, Math.min(M - 1, Math.floor(t)));
      const a = f[j], b = next(j), u = t - j;
      return Math.min(1, (F[j] + (a * u + ((b - a) * u * u) / 2) * dx) / total);
    },
  };
}

// ---------------------------------------------------------------- distance to the normal

/** sup_z |F(μ + σz) − Φ(z)| for a continuous CDF F (the readout "distance to normal"), a numeric estimate: 801 points
 *  on [−8, 8], then golden-section search to 1e-9 in z around every grid local maximum within half of the best one (a
 *  drawn population can have several bumps of nearly the same height: refining only the best grid point could miss
 *  the true supremum, Codex design review 2026-10-07). Beyond ±8 the gap is at most max(F(μ − 8σ), 1 − F(μ + 8σ)) +
 *  Φ(−8), returned as `tail` so a page can tell when the estimate is not the whole story. */
export function distanceToNormal(F: (x: number) => number, mu: number, sd: number): { d: number; z: number; tail: number } {
  const g = (z: number) => Math.abs(F(mu + sd * z) - normalCdf(z));
  const N = 800, lo = -8, step = 16 / N;
  const v = new Float64Array(N + 1);
  let best = 0;
  for (let i = 0; i <= N; i++) { v[i] = g(lo + i * step); if (v[i] > best) best = v[i]; }
  let d = 0, z = 0;
  const R = (Math.sqrt(5) - 1) / 2;
  const search = (a: number, b: number) => {
    let c = b - R * (b - a), e = a + R * (b - a), gc = g(c), ge = g(e);
    while (b - a > 1e-9) {
      if (gc > ge) { b = e; e = c; ge = gc; c = b - R * (b - a); gc = g(c); }
      else { a = c; c = e; gc = ge; e = a + R * (b - a); ge = g(e); }
    }
    for (const t of [a, b, c, e]) { const w = g(t); if (w > d) { d = w; z = t; } }
  };
  for (let i = 0; i <= N; i++) {
    if (v[i] < 0.5 * best || (i > 0 && v[i - 1] > v[i]) || (i < N && v[i + 1] > v[i])) continue;
    if (v[i] > d) { d = v[i]; z = lo + i * step; }
    if (i > 0) search(lo + (i - 1) * step, lo + i * step);
    if (i < N) search(lo + i * step, lo + (i + 1) * step);
  }
  const tail = Math.max(F(mu - 8 * sd), 1 - F(mu + 8 * sd)) + normalCdf(-8);
  return { d, z, tail };
}

// ---------------------------------------------------------------- pointwise bands: exact quantiles, n by n

/**
 * A pointwise band for every n = 0 … M (H12's 95% band, added 2026-10-07): lo[n] is the largest whole a with
 * P(Xₙ < a) ≤ half and hi[n] the smallest whole b with P(Xₙ > b) ≤ half, so P(lo ≤ Xₙ ≤ hi) ≥ 1 − 2·half — the
 * narrowest whole interval with at most `half` outside on each side. `below[n]` = P(Xₙ < lo[n]) and `above[n]` =
 * P(Xₙ > hi[n]) are the two tails left outside (exact quantities, in double precision); the coverage is
 * 1 − below − above. `margin` is how close the closest decision came to `half` (the smallest |P − half| over every
 * comparison that settled an end): when it is far above the rounding error, every end is certainly right. A tail that
 * equals `half` exactly (Binomial(1, 0.05) with alpha 0.1: P(X > 0) = 0.05) is decided by rounding either way and shows
 * as a margin near 0 — a caller relies on the ends only where its margin is clear (H12's p grid: ≥ 3.8e-11, certified
 * by scripts/certify-binomial-band.mjs).
 */
export interface Band { lo: Int32Array; hi: Int32Array; below: Float64Array; above: Float64Array; margin: number }

/** The lower ends for Xₘ ~ Binomial(m, p), m = 0 … M, by walking the quantile: at the current position j it keeps
 *  F = P(Xₘ ≤ j) and f = P(Xₘ = j) > 0; from m to m + 1 (Xₘ₊₁ = Xₘ + a Bernoulli draw) F ← F − p·f and
 *  f ← f·(m + 1)(1 − p)/(m + 1 − j); moving j uses the exact ratio of neighbours (m − j)/(j + 1)·p/(1 − p). The
 *  quantile moves by at most one a step, so the walk is O(M); every operation is a product of positive numbers or one
 *  subtraction of a smaller from a larger probability near the quantile, so nothing underflows or cancels badly. */
function binomialLower(M: number, p: number, half: number) {
  const q = 1 - p, r = p / q;
  const lo = new Int32Array(M + 1), below = new Float64Array(M + 1);
  let margin = Infinity;
  let j = 0, F = 1, f = 1; // m = 0: X = 0
  for (let m = 0; m <= M; m++) {
    if (m > 0) {
      const Fn = F - p * f;
      f = (f * m * q) / (m - j);
      F = Fn;
    }
    // settle j: F(j − 1) = F − f ≤ half < F
    for (;;) {
      if (F <= half) {
        margin = Math.min(margin, half - F);
        f = (f * (m - j) * r) / (j + 1);
        j++;
        F += f;
        continue;
      }
      if (j > 0 && F - f > half) {
        margin = Math.min(margin, F - f - half);
        F -= f;
        f = (f * j) / ((m - j + 1) * r);
        j--;
        continue;
      }
      break;
    }
    margin = Math.min(margin, F - half, j > 0 ? half - (F - f) : Infinity);
    lo[m] = j;
    below[m] = Math.max(0, F - f);
  }
  return { lo, below, margin };
}

/** The exact pointwise band of Binomial(n, p) for n = 0 … M (`Band`); `alpha` is the total outside (0.05: a 95% band).
 *  The upper end comes from the lower end for 1 − p (n − X ~ Binomial(n, 1 − p)), so both tails are computed from
 *  their own side and neither is 1 minus something. About 10 ms for M = 10⁵. */
export function binomialBand(M: number, p: number, alpha = 0.05): Band {
  if (!Number.isInteger(M) || M < 0 || !(p > 0 && p < 1) || !(alpha > 0 && alpha < 1)) throw new RangeError('binomialBand: need a whole M ≥ 0, 0 < p < 1 and 0 < alpha < 1');
  const half = alpha / 2;
  const L = binomialLower(M, p, half), U = binomialLower(M, 1 - p, half);
  const hi = new Int32Array(M + 1);
  for (let n = 0; n <= M; n++) hi[n] = n - U.lo[n];
  return { lo: L.lo, hi, below: L.below, above: U.below, margin: Math.min(L.margin, U.margin) };
}

/**
 * The lower ends of the same band for Sₙ = the sum of n iid copies of a lattice variable with pmf base[k] on
 * k = 0 … s (lattice steps; H12's die is base = [1/6 × 6], S in steps = the total − n), n = 0 … M: lo[n] is the
 * largest a with P(Sₙ < a) ≤ half, below[n] = P(Sₙ < lo[n]). The law of Sₙ is built one copy at a time, keeping only
 * the cells within E[Sₙ] ± (c·s·√n + s): by Hoeffding's inequality P(|Sₙ − E[Sₙ]| ≥ c·s·√n) ≤ 2e^(−2c²), so each step
 * drops at most that much probability (2e⁻⁵⁰ ≈ 4e-22 for c = 5), and a cumulative probability read from the window's
 * left edge is short of the true one by at most the sum of what was dropped (M·2e^(−2c²); 4e-17 for M = 10⁵). The
 * work is about M^1.5·c·s·(s + 1) multiply-adds: seconds for M = 10⁵ — a page ships the result as a table.
 */
export function latticeSumLowerQuantiles(base: ArrayLike<number>, M: number, half: number, c = 5): { lo: Int32Array; below: Float64Array; margin: number } {
  const m = base.length, s = m - 1;
  if (m < 2 || !Number.isInteger(M) || M < 0 || !(half > 0 && half < 0.5)) throw new RangeError('latticeSumLowerQuantiles: need a pmf on two or more points, a whole M ≥ 0 and 0 < half < ½');
  let mean1 = 0;
  for (let k = 0; k < m; k++) mean1 += k * base[k];
  const lo = new Int32Array(M + 1), below = new Float64Array(M + 1);
  let margin = Infinity;
  const width = (n: number) => Math.ceil(c * s * Math.sqrt(n)) + s;
  const cap = 2 * width(M) + 2 * s + 2;
  let cur = new Float64Array(cap), next = new Float64Array(cap);
  let base0 = 0, len = 1; // cur[i] = P(S = base0 + i), i < len
  cur[0] = 1;
  for (let n = 1; n <= M; n++) {
    // convolve: P(Sₙ = t) = Σₖ base[k]·P(Sₙ₋₁ = t − k), over t = base0 … base0 + len − 1 + s
    const full = len + s;
    next.fill(0, 0, full);
    for (let k = 0; k < m; k++) {
      const b = base[k];
      if (!(b > 0)) continue;
      for (let i = 0; i < len; i++) next[i + k] += cur[i] * b;
    }
    // keep the window about the mean
    const mid = n * mean1, w = width(n);
    const keepLo = Math.max(base0, Math.floor(mid - w)), keepHi = Math.min(base0 + full - 1, Math.ceil(mid + w));
    const off = keepLo - base0;
    len = keepHi - keepLo + 1;
    if (off > 0) next.copyWithin(0, off, off + len);
    base0 = keepLo;
    const t = cur; cur = next; next = t;
    // the lower end: the cumulative probability from the window's left edge
    let F = 0, i = 0;
    while (i < len && F + cur[i] <= half) { F += cur[i]; i++; }
    margin = Math.min(margin, half - F, i < len ? F + cur[i] - half : Infinity);
    lo[n] = base0 + i;
    below[n] = F;
  }
  return { lo, below, margin };
}
