// Special functions and the distributions the statistics and calculus pages need: erf / erfc, Φ and its inverse,
// lgamma / gamma, the regularized incomplete gamma and beta functions, and the CDFs, tails and quantiles of the
// normal, t, χ², gamma and beta distributions.
//
// Guarantees (tests/math-special.test.ts checks each against mpmath at 40–60 digits, closed forms and the H20 script):
// - erf / erfc: about 1e-15 relative; erfc keeps its relative accuracy far into the tail (no 1 − erf there, and
//   e^{−x²} is formed without rounding x²), down to the smallest double (erfc(26.5) ≈ 2.2e-307; beyond, 0).
// - Φ and its upper tail Φᶜ: the same, each computed without subtracting from 1 on its small side; dividing z by √2
//   costs about 1e-16·z² relative in the far tail (1e-13 at z = −37).
// - lgamma (x > 0): within 5e-15 absolute where |ln Γ(x)| < 1 — relative near its zeros at 1 and 2 — and within
//   1e-15 relative elsewhere (a 292-point mpmath sweep from 1e-310 to 1e305). Reflection for x ≤ 0.
// - incomplete gamma P, Q and beta I, 1 − I, both tails, against two mpmath grids (gamma: 280 points, shapes 1e-310 to
//   1e7, x down to 1e-300; beta: 380 points, a, b from 1e-20 to 5·10⁵, with points at the switch between the two
//   sides): 1e-12 relative (1e-11 when a beta parameter is below 0.1), or 8 units of the value's own sensitivity to
//   the last bit of x (κ·2⁻⁵³, κ = |d ln F/d ln x|) where that is larger — about 2.4e-12 at a = 10⁷ eight SDs out,
//   where no method can do better from the given x. Results below 2.2e-308 are subnormal and carry the fewer digits
//   a subnormal has. The prefactor xᵃe⁻ˣ/Γ(a) or xᵃ(1−x)ᵇ/B(a, b) is formed from Stirling remainders and
//   ln r − (r − 1) terms, so the large logarithms never cancel (Temme 1979; DiDonato & Morris 1992); the remainder is
//   a series (Numerical Recipes §6.2; DLMF 8.17.8, all terms positive; DLMF 8.7.1 for a small gamma shape) or a
//   continued fraction (Lentz; for beta the odd contraction of DLMF 8.17.22 with its partial denominators formed from
//   1 − x). Each tail is computed directly, never as 1 − the other when the other is near 1.
// - Refusals (NaN, never a truncated number): a shape, scale or degrees of freedom that is not positive and finite; an
//   expansion that does not converge within its 10⁵-term budget — shapes beyond about 10⁸ near their mean
//   (gammaP(1e12, 1e12)), and a beta with a tiny parameter at a small x, where one side is within 1e-3 of 1 (so its
//   complement would lose digits) and the expansion for the other side does not converge (betaI(2e-11, 1e-20, 3));
//   b = 1 and a = 1 are closed forms and always work.
// - t, χ², gamma, beta CDFs and tails inherit those. Quantiles: bisection on the small tail ends at two adjacent
//   doubles and returns the one whose computed probability is nearer the target (betaQuantile(5e-324, 1, 1) is
//   5e-324); a quantile beyond the double range is ±Infinity (t with 1 degree of freedom at 0.9995 is 636.6192487…).
//   The answer is as good as the computed CDF around it: where those probabilities are subnormal (p below 2.2e-308)
//   they are flat over a range of x, and the result is the start of that range (betaQuantile(5e-324, 2, 1) gives
//   1.6e-162, the exact 2⁻⁵³⁷ ≈ 2.2e-162).
//
// Every function is pure and total: arguments outside the domain return NaN (never throw), as the pages turn NaN into
// a message.

const SQRT_PI = Math.sqrt(Math.PI);
const SQRT2 = Math.SQRT2;
const LN_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
const TINY = 1e-300;
/** A usable shape, scale or degrees-of-freedom parameter: positive and finite (anything else gives NaN). */
const pos = (v: number) => v > 0 && v < Infinity;

// ---------------------------------------------------------------- erf, erfc

/** erf by its Maclaurin series; used for |x| < 1, where it converges fast and loses nothing. */
function erfSeries(x: number): number {
  const x2 = x * x;
  let term = x, sum = x;
  for (let n = 1; n < 60; n++) {
    term *= -x2 / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) <= 1e-17 * Math.abs(sum)) break;
  }
  return (2 / SQRT_PI) * sum;
}

/** erfc for x ≥ 1 by its continued fraction (modified Lentz), relative accuracy kept into the far tail. */
function erfcFraction(x: number): number {
  return expNegSquare(x) / SQRT_PI / erfcCF(x);
}

/** The continued fraction f in erfc x = e^{−x²}/(√π f), f = x + (1/2)/(x + 1/(x + (3/2)/(x + …))), for x ≥ 1. */
function erfcCF(x: number): number {
  let f = x, C = x, D = 0;
  for (let k = 1; k < 5000; k++) {
    const a = k / 2;
    D = x + a * D;
    if (Math.abs(D) < TINY) D = TINY;
    C = x + a / C;
    if (Math.abs(C) < TINY) C = TINY;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) <= 3e-16) break;
  }
  return f;
}

/** e^{−x²} without the rounding of x² (Cody 1969): x = h + l with h = x to 1/16, h² exact, x² − h² = (x − h)(x + h). */
function expNegSquare(x: number): number {
  const h = Math.trunc(x * 16) / 16;
  return Math.exp(-h * h) * Math.exp(-(x - h) * (x + h));
}

export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return -erf(-x);
  if (x < 1) return erfSeries(x);
  if (x > 6) return 1; // 1 − erf(6) ≈ 2e-17 is below one ulp of 1
  return 1 - erfcFraction(x);
}

export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return 2 - erfc(-x);
  if (x < 1) return 1 - erfSeries(x); // erfc ≥ 0.157 here: no cancellation worth a digit
  if (x > 27.3) return 0; // below the smallest subnormal
  return erfcFraction(x);
}

// ---------------------------------------------------------------- normal

/** Standard normal density φ(z). */
export const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Φ(z) = P(Z ≤ z), accurate in the lower tail. */
export function normalCdf(z: number): number {
  if (Number.isNaN(z)) return NaN;
  return z < 0 ? 0.5 * erfc(-z / SQRT2) : 1 - 0.5 * erfc(z / SQRT2);
}

/** Φᶜ(z) = P(Z > z) = 1 − Φ(z), accurate in the upper tail. */
export function normalSf(z: number): number {
  if (Number.isNaN(z)) return NaN;
  return z > 0 ? 0.5 * erfc(z / SQRT2) : 1 - 0.5 * erfc(-z / SQRT2);
}

/** P(|Z| > z) for z ≥ 0 (two-sided tail), without cancellation. */
export const normalTwoTail = (z: number) => (z < 0 ? NaN : erfc(z / SQRT2));

/** Acklam's rational approximation (relative error < 1.2e-9), the starting point refined by Halley below. */
function acklam(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/**
 * Standard normal quantile: the z with Φ(z) = p. For an upper-tail probability q (P(Z > z) = q) use
 * `normalQuantileUpper(q)`, which keeps full precision when q is tiny (1 − q would round).
 */
export function normalQuantile(p: number): number {
  if (!(p >= 0 && p <= 1)) return NaN;
  if (p === 0) return -Infinity;
  if (p === 1) return Infinity;
  if (p > 0.5) return -normalQuantileLower(1 - p); // 1 − p is exact for p in (0.5, 1]
  return normalQuantileLower(p);
}

/** The z with P(Z > z) = q. */
export function normalQuantileUpper(q: number): number {
  if (!(q >= 0 && q <= 1)) return NaN;
  if (q === 0) return Infinity;
  if (q === 1) return -Infinity;
  return q <= 0.5 ? -normalQuantileLower(q) : normalQuantileLower(1 - q);
}

/** z with Φ(z) = p for p ≤ 1/2 (z ≤ 0), refined by two Halley steps on the lower tail. */
function normalQuantileLower(p: number): number {
  let x = acklam(p);
  const lnp = Math.log(p);
  for (let i = 0; i < 2; i++) {
    // u = (Φ(x) − p)/φ(x). Below x = −1.5 it is formed as Φ/φ − p/φ, with Φ/φ = 1/(√2 f) from erfc's continued
    // fraction and p/φ = exp(ln p + x²/2 + ln √2π): no e^{x²/2} to overflow and no subnormal Φ(x), so p down to the
    // smallest double works (it gave NaN below about 1e-308)
    const u = x < -1.5
      ? 1 / (SQRT2 * erfcCF(-x / SQRT2)) - Math.exp(lnp + 0.5 * x * x + LN_SQRT_2PI)
      : (normalCdf(x) - p) * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
    x -= u / (1 + 0.5 * x * u);
  }
  return x;
}

// ---------------------------------------------------------------- gamma

/** The Stirling series remainder Σ B₂ₖ / (2k(2k − 1) x^{2k−1}), k = 1 … 7, for x ≥ 10 (error < 1e-17 there). */
function stirlingSeries(x: number): number {
  const x2 = x * x;
  return 1 / (12 * x) - 1 / (360 * x * x2) + 1 / (1260 * x * x2 * x2) - 1 / (1680 * x * x2 ** 3) + 1 / (1188 * x * x2 ** 4)
    - 691 / (360360 * x * x2 ** 5) + 1 / (156 * x * x2 ** 6);
}

const EULER_GAMMA = 0.57721566490153286;
/** ζ(k) for k = 0 … 80 (entries 0 and 1 unused): exact values to 12, then Σ n⁻ᵏ to n = 40 (tail below 1e-19). */
const ZETA: number[] = (() => {
  const z = [NaN, NaN, 1.6449340668482264, 1.2020569031595943, 1.0823232337111382, 1.0369277551433699, 1.0173430619844491,
    1.0083492773819228, 1.0040773561979443, 1.0020083928260822, 1.0009945751278181, 1.0004941886041195, 1.000246086553308];
  for (let k = 13; k <= 80; k++) {
    let sum = 0;
    for (let n = 40; n >= 2; n--) sum += n ** -k;
    z.push(1 + sum);
  }
  return z;
})();

/** ln Γ(1 + e) for |e| ≤ ½: −γe + Σ_{k≥2} ζ(k)(−e)ᵏ/k (DLMF 5.7.3), with full relative accuracy as e → 0. */
function lgamma1pSeries(e: number): number {
  const m = -e;
  let sum = -EULER_GAMMA * e, pw = m;
  for (let k = 2; k <= 80; k++) {
    pw *= m;
    const term = (ZETA[k] * pw) / k;
    sum += term;
    if (Math.abs(term) <= 1e-17 * Math.abs(sum)) break;
  }
  return sum;
}

/** ln Γ(1 + a) for −½ ≤ a ≤ 3/2, accurate relative to its size (it vanishes at a = 0 and a = 1). */
function lgamma1p(a: number): number {
  return a <= 0.5 ? lgamma1pSeries(a) : Math.log(a) + lgamma1pSeries(a - 1); // Γ(1 + a) = aΓ(1 + (a − 1)), a − 1 exact
}

/**
 * ln Γ(x) for x > 0 (and ln|Γ(x)| for negative non-integers): a ζ series on [½, 5/2) (relative accuracy, so the
 * zeros at 1 and 2 are exact), the shifted Stirling series above (about 5e-15 absolute), reflection below ½.
 */
export function lgamma(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return Infinity;
  if (x <= 0) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x); // reflection
  if (x < 0.5) return lgamma1p(x) - Math.log(x); // Γ(x) = Γ(1 + x)/x: exact down to the smallest double
  if (x < 2.5) return lgamma1p(x - 1); // x − 1 is exact here
  let prod = 1;
  while (x < 15) {
    prod *= x; // at most 15 factors below 15: no overflow, one rounding each
    x += 1;
  }
  return (x - 0.5) * Math.log(x) - x + LN_SQRT_2PI + stirlingSeries(x) - Math.log(prod);
}

/** δ(z) = ln Γ(z) − [(z − ½) ln z − z + ½ ln 2π], the small remainder of Stirling's formula (≈ 1/(12z)). */
function stirlingDelta(z: number): number {
  return z >= 10 ? stirlingSeries(z) : lgamma(z) - ((z - 0.5) * Math.log(z) - z + LN_SQRT_2PI);
}

/**
 * ln r − (r − 1) for r = x·k, given ln k: no cancellation near r = 1 (series), and away from it ln r is taken as
 * ln x + ln k, so a subnormal x (or an r that would underflow) keeps its size.
 */
function logRatioTerm(x: number, k: number, lnk: number): number {
  const z = x * k - 1;
  return Math.abs(z) <= 0.5 ? log1pmx(z) : Math.log(x) + lnk - z;
}

/** ln(1 + z) − z for |z| ≤ 1/2, by the series in w = z/(2 + z): ln(1 + z) = 2 atanh w and 2w − z = −z²/(2 + z). */
function log1pmx(z: number): number {
  const w = z / (2 + z), w2 = w * w;
  let term = w * w2, sum = 0;
  for (let k = 3; k < 200; k += 2) {
    const t = term / k;
    sum += t;
    if (Math.abs(t) <= 1e-17 * Math.abs(sum)) break;
    term *= w2;
  }
  return 2 * sum - (z * z) / (2 + z);
}

/** Γ(x). Exact products for positive integers up to 171 (Γ(171) overflows beyond it), else exp(lgamma) with sign. */
export function gamma(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return NaN; // poles
  if (Number.isInteger(x) && x <= 171) {
    let r = 1;
    for (let k = 2; k < x; k++) r *= k;
    return r;
  }
  if (x > 171.7) return Infinity;
  const sign = x > 0 ? 1 : Math.floor(x) % 2 === 0 ? 1 : -1;
  return sign * Math.exp(lgamma(x));
}

/** ln(xᵃe⁻ˣ/Γ(a)) = a(ln r − (r − 1)) + ½ ln(a/2π) − δ(a) with r = x/a: no large terms cancel. */
function gammaLead(a: number, x: number): number {
  // a(ln r − (r − 1)), r = x/a: by the series near r = 1, else as a(ln x − ln a) − (x − a) (no 1/a, which overflows
  // for a tiny shape, and no ln of an r that under- or overflows)
  const z = x / a - 1;
  const f = Math.abs(z) <= 0.5 ? a * log1pmx(z) : a * (Math.log(x) - Math.log(a)) - (x - a);
  return f + 0.5 * Math.log(a / (2 * Math.PI)) - stirlingDelta(a);
}

/**
 * Regularized incomplete gamma: [P(a, x), Q(a, x)] = lower and upper, each computed on its own side. NaN for a shape
 * that is not positive and finite, and when the series or continued fraction does not converge within its budget
 * (10⁵ terms: shapes beyond about 10⁸ near the mean, e.g. gammaP(1e12, 1e12)) — never a truncated value.
 */
export function gammaPQ(a: number, x: number): [number, number] {
  if (!pos(a) || Number.isNaN(x)) return [NaN, NaN];
  if (x <= 0) return [0, 1];
  if (x === Infinity) return [1, 0];
  if (x < a + 1 && a < 0.5) {
    // small shape: with E = xᵃ/Γ(1 + a) and S = Σ_{n≥1} (−x)ⁿ/(n!(a + n)), P = E(1 + aS) (DLMF 8.7.1) and
    // Q = −expm1(ln E) − E·a·S, each side directly (as a → 0, Q ≈ a·E₁(x) and 1 − P would lose it); no 1/a anywhere
    const lnE = a * Math.log(x) - lgamma1p(a);
    let term = 1, S = 0, done = false;
    for (let n = 1; n < 1000; n++) {
      term *= -x / n;
      const add = term / (a + n);
      S += add;
      if (Math.abs(add) <= 1e-17 * Math.abs(S)) {
        done = true;
        break;
      }
    }
    if (!done) return [NaN, NaN];
    const E = Math.exp(lnE);
    return [E * (1 + a * S), -Math.expm1(lnE) - E * a * S];
  }
  const lead = gammaLead(a, x);
  if (x < a + 1) {
    // series Σ xⁿ / (a(a+1)…(a+n)), scaled by a so the first term is 1
    let ap = a, sum = 1, del = 1, done = false;
    for (let n = 0; n < 100000; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-17) {
        done = true;
        break;
      }
    }
    if (!done) return [NaN, NaN];
    const p = (sum / a) * Math.exp(lead);
    return [p, 1 - p];
  }
  // continued fraction for Q: Q = xᵃe⁻ˣ/Γ(a) / F with F = b₀ + a₁/(b₁ + a₂/(b₂ + …)), bⱼ = x + 1 − a + 2j and
  // aⱼ = −j(j − a) (Legendre's fraction, DLMF 8.9.2, in its even contraction), evaluated by modified Lentz (Lentz 1976;
  // Thompson & Barnett 1986) as betacf is. b₀ ≥ 2 here, since x ≥ a + 1.
  let f = x + 1 - a, C = f, D = 0, done = false;
  for (let j = 1; j < 100000; j++) {
    const aj = -j * (j - a), bj = x + 1 - a + 2 * j;
    D = bj + aj * D;
    if (Math.abs(D) < TINY) D = TINY;
    C = bj + aj / C;
    if (Math.abs(C) < TINY) C = TINY;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) <= 3e-16) {
      done = true;
      break;
    }
  }
  if (!done) return [NaN, NaN];
  const q = Math.exp(lead) / f;
  return [1 - q, q];
}

export const gammaP = (a: number, x: number) => gammaPQ(a, x)[0];
export const gammaQ = (a: number, x: number) => gammaPQ(a, x)[1];

// ---------------------------------------------------------------- beta

/**
 * I_x(a, b)·a·B(a, b)/(xᵃyᵇ), y = 1 − x, by the continued fraction of DLMF 8.17.22 in its odd contraction
 * 1/(1 + d₁ − d₁d₂/(1 + d₂ + d₃ − d₃d₄/(1 + d₄ + d₅ − …))), evaluated by modified Lentz, where
 * d₂ₘ = m(b − m)x/((a + 2m − 1)(a + 2m)) and d₂ₘ₊₁ = −(a + m)(a + b + m)x/((a + 2m)(a + 2m + 1)).
 * For large a every d₂ₘ₊₁ is close to −1, so the partial denominators 1 + d₂ₘ + d₂ₘ₊₁ are small differences (the
 * uncontracted form, Numerical Recipes' betacf, loses up to five digits there at a ≈ 10⁵). Here they are formed from y
 * in closed form — 1 + d₁ = (1 − b + (a + b)y)/(a + 1), and for m ≥ 1, with A = a + 2m and s = (d₂ₘ + d₂ₘ₊₁)/x,
 * 1 + s = [A(2m + 1 − b) + (b − 1)(2m + 1) − 2m²]/(A² − 1), so 1 + d₂ₘ + d₂ₘ₊₁ = (1 + s) − s·y — and nothing cancels.
 */
function betacf(x: number, y: number, a: number, b: number): number {
  let f = (1 - b + (a + b) * y) / (a + 1);
  if (Math.abs(f) < TINY) f = TINY;
  let C = f, D = 0;
  for (let m = 1; m <= 50000; m++) {
    const A = a + 2 * m;
    const dPrevOdd = (-(a + m - 1) * (a + b + m - 1) * x) / ((A - 2) * (A - 1)); // d₂ₘ₋₁
    const dEven = (m * (b - m) * x) / ((A - 1) * A); // d₂ₘ
    const s = (m * (b - m)) / ((A - 1) * A) - ((a + m) * (a + b + m)) / (A * (A + 1));
    const den = (A * (2 * m + 1 - b) + (b - 1) * (2 * m + 1) - 2 * m * m) / ((A - 1) * (A + 1)) - s * y;
    const num = -dPrevOdd * dEven;
    D = den + num * D;
    if (Math.abs(D) < TINY) D = TINY;
    C = den + num / C;
    if (Math.abs(C) < TINY) C = TINY;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) <= 3e-16) return 1 / f;
  }
  return NaN; // not converged within the budget: the caller reports NaN rather than a truncated value
}

/**
 * The same quantity as betacf by its power series Σ (a + b)ₙ/(a + 1)ₙ xⁿ (DLMF 8.17.8; TOMS 708's BPSER). Every term is
 * positive, so nothing cancels. Used where it stays short ((a + b)x ≤ 3000, x ≤ 1/2: a few thousand terms at most);
 * that covers a large parameter next to a moderate or small one (the t distribution at ν ≳ 10⁵; a = 100 with
 * b = 5·10⁵), where even the contracted fraction's Lentz steps cancel (measured on the grid: up to 7e-12 with the
 * threshold at 100, 3e-13 at 3000).
 */
function betaSeries(x: number, a: number, b: number): number {
  let term = 1, sum = 1;
  for (let n = 0; n < 100000; n++) {
    term *= ((a + b + n) / (a + 1 + n)) * x;
    sum += term;
    if (term <= 1e-17 * sum) return sum;
  }
  return NaN; // not converged within the budget
}

/** I_x(a, b)·a·B(a, b)/(xᵃ(1 − x)ᵇ): the series where it is short and cancellation-free, else the continued fraction. */
const betaRest = (x: number, y: number, a: number, b: number) => (x <= 0.5 && (a + b) * x <= 3000 ? betaSeries(x, a, b) : betacf(x, y, a, b));

/**
 * ln(xᵃyᵇ/B(a, b)), y = 1 − x, as a(ln r₁ − (r₁ − 1)) + b(ln r₂ − (r₂ − 1)) + ½ ln(ab/(2π(a + b))) + δ(a + b) − δ(a)
 * − δ(b) with r₁ = x(a + b)/a, r₂ = y(a + b)/b: the linear terms a(r₁ − 1) + b(r₂ − 1) are 0 and dropped exactly.
 */
function betaLead(x: number, y: number, a: number, b: number): number {
  const s = a + b;
  return a * logRatioTerm(x, s / a, Math.log1p(b / a)) + b * logRatioTerm(y, s / b, Math.log1p(a / b))
    + 0.5 * Math.log((a / s) * (b / (2 * Math.PI)))
    + stirlingDelta(s) - stirlingDelta(a) - stirlingDelta(b);
}

/**
 * Regularized incomplete beta: [I_x(a, b), 1 − I_x(a, b)], the small one computed directly. Pass `y` = 1 − x when the
 * caller has it more accurately than 1 − x can be formed (x near 1, e.g. ν/(ν + t²) for a large ν).
 */
export function betaI(x: number, a: number, b: number, y = 1 - x): [number, number] {
  if (!pos(a) || !pos(b) || Number.isNaN(x) || Number.isNaN(y)) return [NaN, NaN];
  if (x <= 0) return [0, 1];
  if (y <= 0) return [1, 0];
  // closed forms: I_x(a, 1) = xᵃ and I_x(1, b) = 1 − yᵇ, each side to full precision (a tiny a in particular)
  // (the power itself by Math.pow, within an ulp, rather than exp(a·ln x), which costs |a·ln x|·2⁻⁵³)
  if (b === 1) {
    const l = a * (x < 0.5 ? Math.log(x) : Math.log1p(-y));
    return [x < 0.5 ? Math.pow(x, a) : Math.exp(l), -Math.expm1(l)];
  }
  if (a === 1) {
    const l = b * (y < 0.5 ? Math.log(y) : Math.log1p(-x));
    return [-Math.expm1(l), y < 0.5 ? Math.pow(y, b) : Math.exp(l)];
  }
  const lbt = betaLead(x, y, a, b);
  const lower = () => (Math.exp(lbt) * betaRest(x, y, a, b)) / a; // I, directly
  const upper = () => (Math.exp(lbt) * betaRest(y, x, b, a)) / b; // 1 − I, directly
  // the side the continued fraction converges on fastest first; if it comes out above ¾, the other side is the small
  // one and is computed directly too (a tiny a puts I within 1e-20 of 1: betaI(0.1, 1e-20, 1)), never as 1 − it
  const lowerFirst = x < (a + 1) / (a + b + 2);
  const first = lowerFirst ? lower() : upper();
  if (!(first > 0.75)) return lowerFirst ? [first, 1 - first] : [1 - first, first];
  const second = lowerFirst ? upper() : lower();
  if (second >= 0 && second <= 0.3) return lowerFirst ? [1 - second, second] : [second, 1 - second];
  // the other expansion did not converge: 1 − first is still good to 1e-12 relative while it is above 1e-3
  if (first < 0.999) return lowerFirst ? [first, 1 - first] : [1 - first, first];
  return [NaN, NaN]; // the small side is not computable to its digits: say so
}

export const betaInc = (x: number, a: number, b: number) => betaI(x, a, b)[0];

// ---------------------------------------------------------------- distributions

/** Student t with ν degrees of freedom (ν > 0, not necessarily integer). */
export function tCdf(t: number, nu: number): number {
  return tCdfSf(t, nu)[0];
}
export function tSf(t: number, nu: number): number {
  return tCdfSf(t, nu)[1];
}
/** [P(T ≤ t), P(T > t)], each accurate on its small side, for any finite t. */
function tCdfSf(t: number, nu: number): [number, number] {
  if (!pos(nu) || Number.isNaN(t)) return [NaN, NaN];
  if (t === Infinity) return [1, 0];
  if (t === -Infinity) return [0, 1];
  // the two-sided tail P(|T| > |t|) = I_{ν/(ν+t²)}(ν/2, ½), with both ν/(ν + t²) and t²/(ν + t²) formed directly
  // (never as 1 − the other) and without forming t², which overflows for |t| > 1.3e154
  const at = Math.abs(t);
  let tail2: number;
  if (at <= 1) {
    const t2 = t * t;
    tail2 = betaI(nu / (nu + t2), nu / 2, 0.5, t2 / (nu + t2))[0];
  } else if (Math.log(nu) - 2 * Math.log(at) > -700) {
    const u = nu / at / at;
    tail2 = betaI(u / (1 + u), nu / 2, 0.5, 1 / (1 + u))[0];
  } else {
    // u = ν/t² below 1e-304: I_x(a, ½) = xᵃ/(a·B(a, ½))·(1 + O(x)), exact to double precision
    const a = nu / 2;
    tail2 = Math.exp(a * (Math.log(nu) - 2 * Math.log(at)) - Math.log(a) - (lgamma(a) + lgamma(0.5) - lgamma(a + 0.5)));
  }
  const half = 0.5 * tail2;
  return t >= 0 ? [1 - half, half] : [half, 1 - half];
}

export function tPdf(t: number, nu: number): number {
  if (!pos(nu) || Number.isNaN(t)) return NaN;
  const at = Math.abs(t);
  // ln(1 + t²/ν), without forming t² when it would overflow
  const kernel = ((nu + 1) / 2) * (at > 1e100 ? 2 * Math.log(at) - Math.log(nu) + Math.log1p(nu / at / at) : Math.log1p((t * t) / nu));
  if (nu < 20) return Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - kernel);
  // ln[Γ((ν + 1)/2) / (Γ(ν/2)√(νπ))] = −½ ln 2π + (ν/2)(ln(1 + 1/ν) − 1/ν) + δ((ν + 1)/2) − δ(ν/2): no cancellation
  const norm = -LN_SQRT_2PI + (nu / 2) * log1pmx(1 / nu) + stirlingDelta((nu + 1) / 2) - stirlingDelta(nu / 2);
  return Math.exp(norm - kernel);
}

/** χ² with k degrees of freedom. */
export const chi2Cdf = (x: number, k: number) => (!pos(k) || Number.isNaN(x) ? NaN : x <= 0 ? 0 : gammaP(k / 2, x / 2));
export const chi2Sf = (x: number, k: number) => (!pos(k) || Number.isNaN(x) ? NaN : x <= 0 ? 1 : gammaQ(k / 2, x / 2));
export function chi2Pdf(x: number, k: number): number {
  if (!pos(k) || Number.isNaN(x)) return NaN;
  if (x < 0) return 0;
  if (x === 0) return k < 2 ? Infinity : k === 2 ? 0.5 : 0;
  return gammaPdf(x, k / 2, 2);
}

/** Gamma(shape k, scale θ). */
export const gammaCdf = (x: number, k: number, theta = 1) =>
  !pos(k) || !pos(theta) || Number.isNaN(x) ? NaN : x <= 0 ? 0 : gammaP(k, x / theta);
export const gammaSf = (x: number, k: number, theta = 1) =>
  !pos(k) || !pos(theta) || Number.isNaN(x) ? NaN : x <= 0 ? 1 : gammaQ(k, x / theta);
export function gammaPdf(x: number, k: number, theta = 1): number {
  if (!pos(k) || !pos(theta) || Number.isNaN(x)) return NaN;
  if (x < 0) return 0;
  if (x === 0) return k < 1 ? Infinity : k === 1 ? 1 / theta : 0;
  return Math.exp(gammaLead(k, x / theta)) / x; // (x/θ)ᵏ e^{−x/θ} / Γ(k) / x
}

/** Beta(a, b) on [0, 1]. */
export const betaCdf = (x: number, a: number, b: number) => betaI(x, a, b)[0];
export const betaSf = (x: number, a: number, b: number) => betaI(x, a, b)[1];
export function betaPdf(x: number, a: number, b: number): number {
  if (!pos(a) || !pos(b) || Number.isNaN(x)) return NaN;
  if (x < 0 || x > 1) return 0;
  if (x === 0) return a < 1 ? Infinity : a === 1 ? b : 0;
  if (x === 1) return b < 1 ? Infinity : b === 1 ? a : 0;
  return Math.exp(betaLead(x, 1 - x, a, b)) / (x * (1 - x)); // xᵃ(1 − x)ᵇ / B(a, b) / (x(1 − x))
}

// ---------------------------------------------------------------- quantiles

/**
 * Invert a continuous distribution: the x with cdf(x) = p. Works on the small tail — the lower one through `cdf` when
 * p ≤ 1/2, the upper one through `sf` (with target 1 − p) otherwise — so extreme quantiles keep full precision. The
 * bracket [lo, hi] grows outward (doubling its distance from `center`, never past the largest double) until it holds
 * the target; then bisection runs until lo and hi are adjacent doubles, and the one whose probability is nearer the
 * target is returned. A quantile beyond the double range is ±Infinity; a NaN probability anywhere gives NaN.
 */
function invert(p: number, cdf: (x: number) => number, sf: (x: number) => number, opt: { lo: number; hi: number; center: number; min?: number; max?: number }): number {
  if (!(p >= 0 && p <= 1)) return NaN;
  const min = opt.min ?? -Infinity, max = opt.max ?? Infinity;
  if (p === 0) return min;
  if (p === 1) return max;
  const useUpper = p > 0.5;
  const target = useUpper ? 1 - p : p;
  // g increasing in x
  const g = useUpper ? (x: number) => -sf(x) : cdf;
  const want = useUpper ? -target : target;
  const top = Math.min(max, Number.MAX_VALUE), bottom = Math.max(min, -Number.MAX_VALUE);
  let lo = opt.lo, hi = opt.hi, glo = g(lo), ghi = g(hi);
  for (let i = 0; i < 2000 && !(ghi >= want); i++) {
    if (Number.isNaN(ghi)) return NaN;
    if (hi >= top) return max; // above the largest double
    hi = Math.min(top, opt.center + 2 * (hi - opt.center));
    ghi = g(hi);
  }
  for (let i = 0; i < 2000 && !(glo <= want); i++) {
    if (Number.isNaN(glo)) return NaN;
    if (lo <= bottom) return min;
    lo = Math.max(bottom, opt.center - 2 * (opt.center - lo));
    glo = g(lo);
  }
  for (let i = 0; i < 2200; i++) {
    const mid = Number.isFinite(hi - lo) ? lo + (hi - lo) / 2 : lo / 2 + hi / 2;
    if (mid <= lo || mid >= hi) break;
    const gm = g(mid);
    if (Number.isNaN(gm)) return NaN;
    if (gm < want) {
      lo = mid;
      glo = gm;
    } else {
      hi = mid;
      ghi = gm;
    }
  }
  return Math.abs(ghi - want) <= Math.abs(want - glo) ? hi : lo;
}
export function tQuantile(p: number, nu: number): number {
  if (!pos(nu)) return NaN;
  if (!(p >= 0 && p <= 1)) return NaN;
  if (p === 0.5) return 0; // the symmetric median, exactly
  // Cauchy: −cot(πp) on the small tail (tan(π(p − ½)) would be evaluated at a rounded ±π/2: tQuantile(1e-300, 1)
  // came out as −1.6e16 instead of −3.2e299)
  if (nu === 1) return p === 0 ? -Infinity : p === 1 ? Infinity : p < 0.5 ? -1 / Math.tan(Math.PI * p) : 1 / Math.tan(Math.PI * (1 - p));
  return invert(p, (t) => tCdf(t, nu), (t) => tSf(t, nu), { lo: -1, hi: 1, center: 0 });
}

export function chi2Quantile(p: number, k: number): number {
  if (!pos(k)) return NaN;
  return invert(p, (x) => chi2Cdf(x, k), (x) => chi2Sf(x, k), { lo: 0, hi: Math.max(1, k), center: 0, min: 0 });
}

export function gammaQuantile(p: number, k: number, theta = 1): number {
  if (!pos(k) || !pos(theta)) return NaN;
  return theta * invert(p, (x) => gammaP(k, x), (x) => gammaQ(k, x), { lo: 0, hi: Math.max(1, k), center: 0, min: 0 });
}

export function betaQuantile(p: number, a: number, b: number): number {
  if (!pos(a) || !pos(b)) return NaN;
  return invert(p, (x) => betaCdf(x, a, b), (x) => betaSf(x, a, b), { lo: 0, hi: 1, center: 0.5, min: 0, max: 1 });
}
