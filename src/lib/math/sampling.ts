// Sampling distributions of common statistics (2026-10-08, H18 sampling distribution; H20's intervals from s and H02's
// mean and median may use them). Pure: no DOM. Each law says how it may be called (the batch's principle 8): 'exact' —
// a closed form, evaluated in floating point — or 'numeric' — a quadrature, with the check it passed. Statistics of n
// iid draws X₁ … Xₙ from a population with CDF F and density f:
//
// - the sample variance s² = Σ(x − x̄)²/d (d = n − 1 or n) and s = √s² from a normal population: scaled χ²ₙ₋₁ and its
//   square root (exact); c₄(n) = E[s]/σ for d = n − 1;
// - Var(s²) = σ⁴(2/(n − 1) + κ/n) for any population with a finite fourth moment (κ the excess kurtosis; exact);
// - the k-th order statistic: density betaPdf(F(x); k, n − k + 1)·f(x), CDF betaCdf(F(x); k, n − k + 1) (exact);
//   the median of an odd n is the ((n + 1)/2)-th; of an even n the midpoint of the two middle ones, whose density
//   2∫₀^∞ j(m − t, m + t) dt is a quadrature (numeric), j the joint density of the two;
// - the sum of n Laplace(0, b): a mixture of ±Gamma(n − k, b) with weights C(n − 1 + k, k)/2^(n + k), k < n — every
//   term positive, so it holds at any n (exact);
// - moments of the order statistics of the uniform (Beta) and the exponential (Rényi's representation) (exact);
// - the maximum of a uniform: Beta(n, 1) scaled (exact).
import { betaCdf, betaPdf, chi2Cdf, chi2Pdf, gammaCdf, gammaPdf, lgamma } from './special';
import { gaussLegendreNodes, adaptiveSimpson } from './quad';

export type LawAccuracy = 'exact' | 'numeric';

/** A continuous law: density, CDF, where its mass lies, how it may be called. Numeric laws on a grid also give `mass`
 *  (the grid's total by Simpson's rule before it was divided out: how far the quadrature was from 1) and `moments`
 *  (mean and SD by Simpson's rule on the grid, not from the linear interpolation drawn between its points). */
export interface Law {
  pdf: (x: number) => number;
  cdf: (x: number) => number;
  support: [number, number];
  accuracy: LawAccuracy;
  mass?: number;
  moments?: { mean: number; sd: number };
}

// ---------------------------------------------------------------- the sample variance and SD (normal population)

/** c₄(n) = E[s]/σ for a normal population and s = √(Σ(x − x̄)²/(n − 1)): √(2/(n − 1))·Γ(n/2)/Γ((n − 1)/2). n ≥ 2. */
export function c4(n: number): number {
  if (!(n >= 2)) throw new RangeError('c4: n ≥ 2');
  return Math.sqrt(2 / (n - 1)) * Math.exp(lgamma(n / 2) - lgamma((n - 1) / 2));
}

/** Var(s²) with s² = Σ(x − x̄)²/(n − 1), for any population with variance σ² and excess kurtosis κ (finite fourth
 *  moment): σ⁴(2/(n − 1) + κ/n). n ≥ 2. For the divisor n multiply by ((n − 1)/n)². */
export function varOfS2(sigma: number, kurt: number, n: number): number {
  if (!(n >= 2)) throw new RangeError('varOfS2: n ≥ 2');
  return sigma ** 4 * (2 / (n - 1) + kurt / n);
}

/** The law of Σ(x − x̄)²/d for n draws from N(μ, σ²): (σ²/d)·χ²ₙ₋₁. */
export function normalVarianceLaw(sigma: number, n: number, d: number): Law {
  if (!(n >= 2) || !(d > 0)) throw new RangeError('normalVarianceLaw: n ≥ 2, d > 0');
  const c = (sigma * sigma) / d, k = n - 1;
  // at v = 0 the density is chi2Pdf's own limit: +∞ for n = 2 (one degree of freedom), 1/(2c) for n = 3, else 0
  return { pdf: (v) => (v < 0 ? 0 : chi2Pdf(v / c, k) / c), cdf: (v) => (v <= 0 ? 0 : chi2Cdf(v / c, k)), support: [0, Infinity], accuracy: 'exact' };
}

/** The law of √(Σ(x − x̄)²/d) for n draws from N(μ, σ²): a scaled χ-distribution with n − 1 degrees of freedom. At
 *  t = 0 the density is its limit, not 0·∞: √(2/(πc)) for n = 2 (a half-normal), 0 for n ≥ 3 (c = σ²/d). */
export function normalSdLaw(sigma: number, n: number, d: number): Law {
  const v = normalVarianceLaw(sigma, n, d), c = (sigma * sigma) / d;
  const at0 = n === 2 ? Math.sqrt(2 / (Math.PI * c)) : 0;
  return { pdf: (t) => (t < 0 ? 0 : t === 0 ? at0 : 2 * t * v.pdf(t * t)), cdf: (t) => (t <= 0 ? 0 : v.cdf(t * t)), support: [0, Infinity], accuracy: 'exact' };
}

/** Mean and SD of √(Σ(x − x̄)²/d) for a normal population: E = σ·c₄(n)·√((n − 1)/d), SD = √(σ²(n − 1)/d − E²). */
export function normalSdMoments(sigma: number, n: number, d: number): { mean: number; sd: number } {
  const m = sigma * c4(n) * Math.sqrt((n - 1) / d);
  return { mean: m, sd: Math.sqrt(Math.max(0, (sigma * sigma * (n - 1)) / d - m * m)) };
}

// ---------------------------------------------------------------- order statistics

/** The k-th smallest of n draws (1 ≤ k ≤ n) from a continuous population: exact given F and f. */
export function orderStatLaw(F: (x: number) => number, f: (x: number) => number, n: number, k: number, support: [number, number]): Law {
  if (!Number.isInteger(n) || !Number.isInteger(k) || k < 1 || k > n) throw new RangeError('orderStatLaw: 1 ≤ k ≤ n, whole numbers');
  const a = k, b = n - k + 1;
  return {
    pdf: (x) => {
      const fx = f(x);
      if (!(fx > 0)) return 0;
      return betaPdf(Math.min(1, Math.max(0, F(x))), a, b) * fx;
    },
    cdf: (x) => {
      const u = F(x);
      return u <= 0 ? 0 : u >= 1 ? 1 : betaCdf(u, a, b);
    },
    support,
    accuracy: 'exact',
  };
}

/** What a population gives the median's law: its CDF, survival function (for the upper tail without cancellation),
 *  density, where its mass lies, a finite range covering all but a negligible part of it, and the points where its
 *  density has a kink or jump (the quadrature splits there). */
export interface PopulationLaw {
  F: (x: number) => number;
  S?: (x: number) => number;
  f: (x: number) => number;
  support: [number, number];
  /** a finite range holding all but ~1e-10 of the population */
  range: [number, number];
  kinks?: number[];
}

/** The density of the midpoint of the two middle order statistics of an even n at m: 2∫₀^T j(m − t, m + t) dt with
 *  j(u, v) = n!/((k − 1)!)²·F(u)^{k−1}·S(v)^{k−1}·f(u)·f(v), k = n/2. Gauss–Legendre (`nodes` points) on panels that
 *  double in width from every break (t = 0, each kink's |m − c|), so a peak at t = 0 as narrow as the range × 2⁻²⁰ is
 *  resolved and the long tail costs a few panels. Breaks closer than the floating-point grain at m are one break, and
 *  every panel moves on (Codex implementation review 2026-10-13: a panel narrower than an ulp never advanced). */
function midpointDensity(p: PopulationLaw, n: number, m: number, nodes = 8): number {
  const k = n / 2;
  const lnC = Math.log(2) + lgamma(n + 1) - 2 * lgamma(k);
  const S = p.S ?? ((x: number) => 1 - p.F(x));
  const [lo, hi] = p.range;
  const T = Math.min(m - Math.max(lo, p.support[0]), Math.min(hi, p.support[1]) - m);
  if (!(T > 0)) return 0;
  const g = (t: number) => {
    const u = m - t, v = m + t;
    const fu = p.f(u), fv = p.f(v);
    if (!(fu > 0) || !(fv > 0)) return 0;
    const Fu = p.F(u), Sv = S(v);
    if (k > 1 && (!(Fu > 0) || !(Sv > 0))) return 0;
    return Math.exp(lnC + (k > 1 ? (k - 1) * (Math.log(Fu) + Math.log(Sv)) : 0)) * fu * fv;
  };
  // u = m − t and v = m + t are resolved only to the grain of m: breaks closer than that are the same break
  const grain = 64 * Number.EPSILON * Math.max(1, Math.abs(m) + T);
  const breaks = [0];
  for (const t of [...(p.kinks ?? []).map((c) => Math.abs(m - c)), T].sort((a, b) => a - b)) {
    if (t > breaks[breaks.length - 1] + grain && t <= T) breaks.push(t);
  }
  if (breaks[breaks.length - 1] < T) breaks[breaks.length - 1] = T;
  const { x: gx, w: gw } = gaussLegendreNodes(nodes);
  let sum = 0;
  for (let i = 0; i + 1 < breaks.length; i++) {
    const a = breaks[i], len = breaks[i + 1] - a;
    if (!(len > 0)) continue;
    // panels [a, a + h], [a + h, a + 2h], [a + 2h, a + 4h] …, the last ending at a + len — from t = 0, where a narrow
    // peak can sit; between two kinks further out (a drawn polyline's corners) the integrand is smooth: one panel
    let x0 = a, w = Math.max(i === 0 || breaks.length < 12 ? len * 2 ** -20 : len, grain);
    for (let panels = 0; x0 < a + len && panels < 200; panels++) {
      const x1 = Math.min(a + len, x0 + w);
      if (!(x1 > x0)) break;
      const c = (x0 + x1) / 2, r = (x1 - x0) / 2;
      for (let j = 0; j < gx.length; j++) sum += gw[j] * r * g(c + r * gx[j]);
      if (x0 > a) w *= 2;
      x0 = x1;
    }
  }
  return sum;
}

/**
 * The law of the sample median of n draws. Odd n: the ((n + 1)/2)-th order statistic, exact. Even n: the midpoint of
 * the n/2-th and (n/2 + 1)-th, numeric — its density on `points` grid points (each a quadrature, `midpointDensity`),
 * linear between them, the CDF the cumulative trapezoid of that polyline, both divided by its total `mass`. The grid
 * spans where the midpoint can be: it lies between the two order statistics, so below the k-th's 1e-10 quantile or
 * above the (k + 1)-th's 1 − 1e-10 quantile it has less than 1e-10 of its mass. The page calls it "approximate" when
 * |mass − 1| > 1e-6.
 */
export function medianLaw(p: PopulationLaw, n: number, points = 601, nodes = 8): Law {
  return runSteps(medianLawSteps(p, n, points, nodes));
}

/** Run a computation written in steps (below) to its end at once. */
export function runSteps<T>(g: Generator<void, T>): T {
  let r = g.next();
  while (!r.done) r = g.next();
  return r.value;
}

/** `medianLaw` in steps: it yields after each grid point (each a quadrature), so a page can spread the work over
 *  frames and drop it when the reader moves on (Codex implementation review 2026-10-13: ~50 ms at n = 200 in one go). */
export function* medianLawSteps(p: PopulationLaw, n: number, points = 601, nodes = 8): Generator<void, Law> {
  if (points % 2 === 0) points++; // Simpson's rule needs an even number of intervals
  if (!Number.isInteger(n) || n < 1) throw new RangeError('medianLaw: n a whole number ≥ 1');
  const sup: [number, number] = [Math.max(p.support[0], p.range[0]), Math.min(p.support[1], p.range[1])];
  if (n % 2 === 1) return orderStatLaw(p.F, p.f, n, (n + 1) / 2, p.support);
  const k = n / 2;
  const lower = orderStatLaw(p.F, p.f, n, k, p.support), upper = orderStatLaw(p.F, p.f, n, k + 1, p.support);
  const lo = Math.max(sup[0], quantileOf(lower.cdf, 1e-10, sup[0], sup[1]));
  const hi = Math.min(sup[1], quantileOf(upper.cdf, 1 - 1e-10, sup[0], sup[1]));
  const dx = (hi - lo) / (points - 1);
  const xs = new Float64Array(points), fs = new Float64Array(points);
  for (let i = 0; i < points; i++) { xs[i] = lo + i * dx; fs[i] = midpointDensity(p, n, xs[i], nodes); yield; }
  const cum = new Float64Array(points);
  for (let i = 1; i < points; i++) cum[i] = cum[i - 1] + ((fs[i - 1] + fs[i]) / 2) * dx;
  const total = cum[points - 1] > 0 ? cum[points - 1] : 1;
  // Simpson's rule on the grid (an even number of intervals) for the mass and the moments
  const simpson = (g: (i: number) => number) => {
    let t = g(0) + g(points - 1);
    for (let i = 1; i < points - 1; i++) t += (i % 2 ? 4 : 2) * g(i);
    return (t * dx) / 3;
  };
  const mass = simpson((i) => fs[i]);
  const mean = simpson((i) => xs[i] * fs[i]) / mass;
  const sd = Math.sqrt(simpson((i) => (xs[i] - mean) ** 2 * fs[i]) / mass);
  const at = (x: number) => Math.min(points - 2, Math.max(0, Math.floor((x - lo) / dx)));
  return {
    pdf: (x) => {
      if (!(x >= lo && x <= hi)) return 0;
      const i = at(x), u = (x - xs[i]) / dx;
      return (fs[i] + (fs[i + 1] - fs[i]) * u) / total;
    },
    cdf: (x) => {
      if (x <= lo) return 0;
      if (x >= hi) return 1;
      const i = at(x), u = (x - xs[i]) / dx, a = fs[i], b = fs[i + 1];
      return Math.min(1, (cum[i] + (a * u + ((b - a) * u * u) / 2) * dx) / total);
    },
    support: [lo, hi],
    accuracy: 'numeric',
    mass,
    moments: { mean, sd },
  };
}

/**
 * The refinement check of an even n's numeric median law (Codex design review 2026-10-08: the mass alone cannot vouch
 * for the shape): the law again on a grid twice as fine with twice the quadrature nodes in each density (the inner
 * integral refined too: Codex implementation review 2026-10-13), the largest difference between the two CDFs at the
 * coarse grid's points, and the differences of the moments. The tails need no second look: the grid spans all but
 * 2·10⁻¹⁰ of the law, and anything lost there or in the quadrature shows in the raw mass. The page shows the law only
 * when `lawPasses`. Costs four laws: run it in steps (`medianLawCheckSteps`).
 */
export function medianLawCheck(p: PopulationLaw, n: number, law: Law, points = 601): { cdf: number; mean: number; sd: number } {
  return runSteps(medianLawCheckSteps(p, n, law, points));
}
export function* medianLawCheckSteps(p: PopulationLaw, n: number, law: Law, points = 601): Generator<void, { cdf: number; mean: number; sd: number }> {
  if (n % 2 === 1 || !law.moments) return { cdf: 0, mean: 0, sd: 0 };
  const fine = yield* medianLawSteps(p, n, 2 * points - 1, 16);
  const [lo, hi] = law.support;
  let d = 0;
  for (let i = 0; i < points; i++) { const x = lo + ((hi - lo) * i) / (points - 1); d = Math.max(d, Math.abs(law.cdf(x) - fine.cdf(x))); }
  const m = fine.moments!;
  return { cdf: d, mean: Math.abs(law.moments.mean - m.mean) / Math.max(1, Math.abs(m.mean)), sd: Math.abs(law.moments.sd - m.sd) / m.sd };
}

/** Whether a numeric law passes: finite raw mass within 1e-6 of 1 and, given a check, everything within 1e-4. */
export function lawPasses(law: Law, check?: { cdf: number; mean: number; sd: number }): boolean {
  if (law.accuracy === 'exact') return true;
  if (law.mass !== undefined && !(Math.abs(law.mass - 1) <= 1e-6)) return false;
  return !check || (check.cdf <= 1e-4 && check.mean <= 1e-4 && check.sd <= 1e-4);
}

/** Mean and SD of a law by quadrature over [lo, hi], split at `kinks` (adaptive Simpson, tolerance 1e-12 per piece):
 *  numeric. `mass` is ∫pdf over the range (it should be 1). */
export function lawMoments(law: Pick<Law, 'pdf'>, lo: number, hi: number, kinks: number[] = []): { mean: number; sd: number; mass: number } {
  const cuts = [lo, ...kinks.filter((c) => c > lo && c < hi).sort((a, b) => a - b), hi];
  let m0 = 0, m1 = 0, m2 = 0;
  for (let i = 0; i + 1 < cuts.length; i++) {
    const a = cuts[i], b = cuts[i + 1];
    m0 += adaptiveSimpson(law.pdf, a, b, { tol: 1e-12, minDepth: 8 }).value;
    m1 += adaptiveSimpson((x) => x * law.pdf(x), a, b, { tol: 1e-12, minDepth: 8 }).value;
  }
  const mean = m1 / m0;
  for (let i = 0; i + 1 < cuts.length; i++) m2 += adaptiveSimpson((x) => (x - mean) ** 2 * law.pdf(x), cuts[i], cuts[i + 1], { tol: 1e-12, minDepth: 8 }).value;
  return { mean, sd: Math.sqrt(m2 / m0), mass: m0 };
}

/** The p-quantile of a continuous CDF by bisection in [lo, hi] (F(lo) ≤ p ≤ F(hi)). */
export function quantileOf(F: (x: number) => number, p: number, lo: number, hi: number): number {
  let a = lo, b = hi;
  for (let i = 0; i < 200 && b - a > 1e-12 * Math.max(1, Math.abs(a) + Math.abs(b)); i++) {
    const m = (a + b) / 2;
    if (F(m) < p) a = m;
    else b = m;
  }
  return (a + b) / 2;
}

// ---------------------------------------------------------------- order statistics with closed-form moments

/** E and Var of the k-th of n draws from U(a, b): a + (b − a)·Beta(k, n + 1 − k). */
export function uniformOrderMoments(n: number, k: number, a: number, b: number): { mean: number; variance: number } {
  const L = b - a;
  return { mean: a + (L * k) / (n + 1), variance: (L * L * k * (n + 1 - k)) / ((n + 1) ** 2 * (n + 2)) };
}

/** Mean and SD of the sample median of n draws from U(a, b) (exact): odd n the middle order statistic; even n the
 *  midpoint, with Cov(U_(i), U_(j)) = i(n + 1 − j)/((n + 1)²(n + 2)) for i ≤ j (scaled by (b − a)²). */
export function uniformMedianMoments(n: number, a: number, b: number): { mean: number; sd: number } {
  const L = b - a;
  if (n % 2 === 1) { const o = uniformOrderMoments(n, (n + 1) / 2, a, b); return { mean: o.mean, sd: Math.sqrt(o.variance) }; }
  const k = n / 2, d = (n + 1) ** 2 * (n + 2);
  const vk = (k * (n + 1 - k)) / d, vk1 = ((k + 1) * (n - k)) / d, cov = (k * (n - k)) / d;
  return { mean: (a + b) / 2, sd: L * Math.sqrt((vk + vk1 + 2 * cov) / 4) };
}

/** E and Var of the k-th of n draws from shift + θ·Exp(1), by Rényi's representation: the k-th is
 *  shift + θ·Σᵢ₌ₙ₋ₖ₊₁ⁿ Eᵢ/i with independent Exp(1) Eᵢ. */
export function expoOrderMoments(n: number, k: number, theta: number, shift = 0): { mean: number; variance: number } {
  let s1 = 0, s2 = 0;
  for (let i = n - k + 1; i <= n; i++) { s1 += 1 / i; s2 += 1 / (i * i); }
  return { mean: shift + theta * s1, variance: theta * theta * s2 };
}

/** Mean and SD of the sample median of n draws from shift + θ·Exp(1) (exact). Even n: the (k + 1)-th is the k-th plus
 *  θ·E/(n − k), independent of it (Rényi), so the midpoint is the k-th plus θ·E/(2(n − k)). */
export function expoMedianMoments(n: number, theta: number, shift = 0): { mean: number; sd: number } {
  if (n % 2 === 1) { const o = expoOrderMoments(n, (n + 1) / 2, theta, shift); return { mean: o.mean, sd: Math.sqrt(o.variance) }; }
  const k = n / 2, o = expoOrderMoments(n, k, theta, shift), g = theta / (2 * (n - k));
  return { mean: o.mean + g, sd: Math.sqrt(o.variance + g * g) };
}

// ---------------------------------------------------------------- the maximum of a uniform

/** The largest of n draws from U(a, b): (b − a)·Beta(n, 1) + a. Exact. */
export function uniformMaxLaw(n: number, a: number, b: number): Law {
  const L = b - a;
  return {
    pdf: (x) => (x < a || x > b ? 0 : (n / L) * ((x - a) / L) ** (n - 1)),
    cdf: (x) => (x <= a ? 0 : x >= b ? 1 : ((x - a) / L) ** n),
    support: [a, b],
    accuracy: 'exact',
  };
}

/** Mean and SD of the largest of n draws from U(a, b): a + (b − a)·n/(n + 1), (b − a)·√(n/((n + 1)²(n + 2))). */
export function uniformMaxMoments(n: number, a: number, b: number): { mean: number; sd: number } {
  const L = b - a;
  return { mean: a + (L * n) / (n + 1), sd: L * Math.sqrt(n / ((n + 1) ** 2 * (n + 2))) };
}

// ---------------------------------------------------------------- sums of Laplace draws

/** The weights of the sum of n Laplace(0, b) as a symmetric mixture of Gamma(n − k, b) magnitudes:
 *  w_k = C(n − 1 + k, k)/2^(n + k), k = 0 … n − 1; they add up to ½ (each side). */
export function laplaceSumWeights(n: number): Float64Array {
  if (!Number.isInteger(n) || n < 1) throw new RangeError('laplaceSumWeights: n a whole number ≥ 1');
  const w = new Float64Array(n);
  for (let k = 0; k < n; k++) w[k] = Math.exp(lgamma(n + k) - lgamma(k + 1) - lgamma(n) - (n + k) * Math.LN2);
  return w;
}

/** The law of X₁ + … + Xₙ for iid Laplace(0, b): f(s) = Σₖ wₖ·gammaPdf(|s|; n − k, b), F(s) = ½ ± Σₖ wₖ·gammaCdf(|s|;
 *  n − k, b). Exact (every term positive: no cancellation at any n). */
export function laplaceSumLaw(n: number, b: number): Law {
  const w = laplaceSumWeights(n);
  const pdf = (s: number) => {
    const a = Math.abs(s);
    let t = 0;
    for (let k = 0; k < n; k++) if (w[k] > 1e-300) t += w[k] * gammaPdf(a, n - k, b);
    return t;
  };
  const half = (a: number) => {
    let t = 0;
    for (let k = 0; k < n; k++) if (w[k] > 1e-300) t += w[k] * gammaCdf(a, n - k, b);
    return t;
  };
  // n = 1: the Laplace density itself has its peak (a kink) at 0; Gamma(1, b) at 0 is 1/b, so f(0) = 1/(2b)
  return { pdf, cdf: (s) => (s >= 0 ? 0.5 + half(s) : 0.5 - half(-s)), support: [-Infinity, Infinity], accuracy: 'exact' };
}
