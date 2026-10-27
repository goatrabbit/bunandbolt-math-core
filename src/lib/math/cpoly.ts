// Complex polynomials in double precision (2026-10-11, WB03 Newton fractal). A number is a pair [re, im]; a polynomial
// is its coefficients from the constant term up, p(z) = Σ c[k] zᵏ. What this file promises:
// - fromRoots, evaluate, derivative and taylorAt are the plain algebra, exact up to rounding (Horner's scheme, repeated
//   synthetic division for the Taylor coefficients b[k] = p⁽ᵏ⁾(ζ)/k!).
// - rootsOf (Durand–Kerner, also called Weierstrass) returns all n roots of a degree-n polynomial with ok: true when the
//   last correction was below the tolerance. Simple roots come to about 1e-15 relative; a k-fold root only to about
//   eps^(1/k) (1e-8 for a double root) — the method's own limit, not a bug.
// - smaleGamma / smaleRadius: Smale's γ(ζ) = max_{k≥2} |p⁽ᵏ⁾(ζ) / (k! p′(ζ))|^{1/(k−1)} at a simple root ζ, and
//   (3 − √7)/(2γ): every start closer to ζ than this is an approximate zero of p with associated zero ζ — Newton's
//   method from it converges to ζ, |z_k − ζ| ≤ (1/2)^(2^k − 1)|z₀ − ζ| (Smale, "Newton's method estimates from data at
//   one point", 1986; Blum, Cucker, Shub, Smale, Complexity and Real Computation, 1998, ch. 8). Not defined at a
//   multiple root (p′(ζ) = 0): γ is then Infinity and the radius 0. These start from expanded coefficients, so for roots
//   close together cancellation can make γ far too small (roots 3, 3.01, …, 3.07: a radius 3 times too large). When
//   the roots are what is known, use the root forms below.
// - taylorRatiosAtRoot / smaleGammaAtRoot / smaleRadiusAtRoot: the same quantities at a root of p = Π (z − rⱼ), straight
//   from the root differences: A_k = p⁽ᵏ⁾(ζ)/(k! p′(ζ)) is the elementary symmetric sum e_{k−1} of the 1/(ζ − rⱼ), j ≠ i,
//   because p(ζ + e)/p′(ζ) = e·Π_{j≠i} (1 + e/(ζ − rⱼ)). No cancellation between coefficients; the rounding left is
//   bounded by the same sums over |1/(ζ − rⱼ)| times a generous multiple of the unit roundoff, and γ is rounded up by
//   that bound, so the radius is certified for the roots as given (doubles).
// - rouche: whether a polynomial given by its Taylor coefficients about c has exactly m zeros within ρ of c (Rouché's
//   theorem with |t_m|ρ^m > Σ_{j≠m} |t_j|ρ^j; up to the rounding of the coefficients).
// Tests: tests/math-cpoly.test.ts.
export type Z = [re: number, im: number];

export const cadd = (a: Z, b: Z): Z => [a[0] + b[0], a[1] + b[1]];
export const csub = (a: Z, b: Z): Z => [a[0] - b[0], a[1] - b[1]];
export const cmul = (a: Z, b: Z): Z => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
export function cdiv(a: Z, b: Z): Z {
  const q = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / q, (a[1] * b[0] - a[0] * b[1]) / q];
}
export const cabs = (a: Z): number => Math.hypot(a[0], a[1]);
export const carg = (a: Z): number => Math.atan2(a[1], a[0]);
/** a^p for a real p, principal branch (arg in (−π, π]). */
export function cpow(a: Z, p: number): Z {
  const r = cabs(a) ** p, t = carg(a) * p;
  return r === 0 ? [0, 0] : [r * Math.cos(t), r * Math.sin(t)];
}

/** The monic polynomial with these roots (repeated roots repeated), coefficients from the constant term up. */
export function fromRoots(roots: readonly Z[]): Z[] {
  let c: Z[] = [[1, 0]];
  for (const r of roots) {
    const n: Z[] = Array.from({ length: c.length + 1 }, () => [0, 0] as Z);
    for (let k = 0; k < c.length; k++) { n[k + 1] = cadd(n[k + 1], c[k]); n[k] = csub(n[k], cmul(r, c[k])); }
    c = n;
  }
  return c;
}
/** p(z) by Horner's scheme. */
export function evaluate(c: readonly Z[], z: Z): Z {
  let s: Z = [0, 0];
  for (let k = c.length - 1; k >= 0; k--) s = cadd(cmul(s, z), c[k]);
  return s;
}
export const derivative = (c: readonly Z[]): Z[] => c.slice(1).map((x, k) => [x[0] * (k + 1), x[1] * (k + 1)] as Z);
/** b[k] = p⁽ᵏ⁾(ζ)/k!, k = 0 … deg p: the coefficients of p(ζ + e) in e. */
export function taylorAt(c: readonly Z[], zeta: Z): Z[] {
  let a = c.map((x) => [x[0], x[1]] as Z);
  const b: Z[] = [];
  while (a.length) {
    // one pass of synthetic division by (z − ζ): the remainder is the value, the quotient goes on
    const acc: Z[] = new Array(a.length);
    let s: Z = [0, 0];
    for (let j = a.length - 1; j >= 0; j--) { s = cadd(cmul(s, zeta), a[j]); acc[j] = s; }
    b.push(acc[0]);
    a = acc.slice(1);
  }
  return b;
}

/** All roots of a polynomial of degree ≥ 1 (Durand–Kerner from the points 0.4 + 0.9i to the k-th power, scaled). */
export function rootsOf(c: readonly Z[], o: { maxIter?: number; tol?: number } = {}): { roots: Z[]; ok: boolean } {
  let n = c.length - 1;
  while (n > 0 && c[n][0] === 0 && c[n][1] === 0) n--;
  if (n < 1) return { roots: [], ok: false };
  const lead = c[n], m = c.slice(0, n + 1).map((x) => cdiv(x, lead));
  // starting points on a circle of the Cauchy bound's size
  let R = 0;
  for (let k = 0; k < n; k++) R = Math.max(R, cabs(m[k]));
  const scale = Math.min(1 + R, 1e6) * 0.5 + 0.5;
  let r: Z[] = Array.from({ length: n }, (_, i) => { const w = cpow([0.4, 0.9], i); return [w[0] * scale, w[1] * scale] as Z; });
  const tol = o.tol ?? 1e-15, maxIter = o.maxIter ?? 800;
  for (let it = 0; it < maxIter; it++) {
    let worst = 0;
    r = r.map((z, i) => {
      let den: Z = [1, 0];
      for (let j = 0; j < n; j++) if (j !== i) den = cmul(den, csub(z, r[j]));
      const dz = cdiv(evaluate(m, z), den);
      worst = Math.max(worst, cabs(dz) / Math.max(1, cabs(z)));
      return csub(z, dz);
    });
    if (!(worst > tol)) return { roots: r, ok: Number.isFinite(worst) };
  }
  return { roots: r, ok: false };
}

/** Smale's γ at ζ (Infinity where p′(ζ) = 0). */
export function smaleGamma(c: readonly Z[], zeta: Z): number {
  const b = taylorAt(c, zeta), b1 = cabs(b[1]);
  if (b1 === 0) return Infinity;
  let g = 0;
  for (let k = 2; k < b.length; k++) g = Math.max(g, (cabs(b[k]) / b1) ** (1 / (k - 1)));
  return g;
}
/** The radius (3 − √7)/(2γ) of approximate zeros around a simple root (0 at a multiple root; Infinity for degree 1). */
export const SMALE_ALPHA = (3 - Math.sqrt(7)) / 2;
export function smaleRadius(c: readonly Z[], zeta: Z): number {
  const g = smaleGamma(c, zeta);
  return g === 0 ? Infinity : SMALE_ALPHA / g;
}

/** The unit roundoff of doubles. */
const UNIT = 2 ** -53;
/** At the root ζ = roots[i] of Π (z − rⱼ): a[k] = A_k = p⁽ᵏ⁾(ζ)/(k! p′(ζ)) for k = 1 … n (a[0] unused, A_1 = 1), from
 *  the root differences; err[k] bounds the rounding error of a[k]. */
export function taylorRatiosAtRoot(roots: readonly Z[], i: number): { a: Z[]; err: number[] } {
  const n = roots.length;
  const a: Z[] = Array.from({ length: n + 1 }, () => [0, 0] as Z);
  const mag: number[] = new Array(n + 1).fill(0);
  a[1] = [1, 0]; mag[1] = 1;
  let top = 1;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    // one more factor (1 + e·w): the elementary symmetric sums of the w's so far, from the top down
    const w = cdiv([1, 0], csub(roots[i], roots[j])), aw = cabs(w);
    for (let k = top + 1; k >= 2; k--) { a[k] = cadd(a[k], cmul(w, a[k - 1])); mag[k] += aw * mag[k - 1]; }
    top++;
  }
  return { a, err: mag.map((m) => 16 * (n + 2) * UNIT * m) };
}
/** Smale's γ at roots[i], rounded up (Infinity if another root coincides with it). */
export function smaleGammaAtRoot(roots: readonly Z[], i: number): number {
  const { a, err } = taylorRatiosAtRoot(roots, i);
  let g = 0;
  for (let k = 2; k < a.length; k++) g = Math.max(g, (cabs(a[k]) + err[k]) ** (1 / (k - 1)));
  return Number.isFinite(g) ? g * (1 + 1e-12) : Infinity;
}
/** (3 − √7)/(2γ) at roots[i], rounded down: certified for the roots as given. */
export function smaleRadiusAtRoot(roots: readonly Z[], i: number): number {
  const g = smaleGammaAtRoot(roots, i);
  return g === 0 ? Infinity : (SMALE_ALPHA / g) * (1 - 1e-12);
}
/** Rouché's theorem on the Taylor coefficients t about c: exactly m zeros within ρ of c. */
export function rouche(t: readonly Z[], m: number, rho: number): boolean {
  if (m >= t.length) return false;
  let rest = 0;
  for (let j = 0; j < t.length; j++) if (j !== m) rest += cabs(t[j]) * rho ** j;
  return cabs(t[m]) * rho ** m > rest * (1 + 1e-9);
}
