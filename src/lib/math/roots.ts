// Root finding. Every function here needs a bracket (a sign change) or a description of where the lobes are; none
// of them guesses. A plain scan of a fixed grid can step over a root between two samples — the D01 discussion
// reproduced it for x·sin(1/x) near 0 — so oscillating functions go through `lobeCrossings`, which is told where the
// oscillation's lobes begin and end.

export interface Root {
  x: number;
  /** false when [a, b] did not bracket a sign change (x is then NaN) or the budget ran out */
  ok: boolean;
  iterations: number;
}

/** a + (b − a)/2 without overflowing when b − a exceeds the double range (−1e308 … 1e308). */
const midpoint = (a: number, b: number) => (Number.isFinite(b - a) ? a + (b - a) / 2 : a / 2 + b / 2);

/**
 * Bisection on a bracket [a, b] with f(a)·f(b) ≤ 0, down to adjacent doubles (or `xtol`). A NaN from f anywhere ends
 * it with ok: false (it cannot tell which side the root is on).
 */
export function bisect(f: (x: number) => number, a: number, b: number, opt: { xtol?: number; maxIter?: number } = {}): Root {
  let fa = f(a), fb = f(b);
  if (fa === 0) return { x: a, ok: true, iterations: 0 };
  if (fb === 0) return { x: b, ok: true, iterations: 0 };
  if (!(Math.sign(fa) * Math.sign(fb) < 0)) return { x: NaN, ok: false, iterations: 0 };
  const xtol = opt.xtol ?? 0, maxIter = opt.maxIter ?? 2000;
  let i = 0;
  for (; i < maxIter; i++) {
    const m = midpoint(a, b);
    if (m === a || m === b || Math.abs(b - a) <= xtol) break;
    const fm = f(m);
    if (fm === 0) return { x: m, ok: true, iterations: i + 1 };
    if (Number.isNaN(fm)) return { x: NaN, ok: false, iterations: i + 1 };
    if (Math.sign(fm) === Math.sign(fa)) {
      a = m;
      fa = fm;
    } else {
      b = m;
      fb = fm;
    }
  }
  return { x: midpoint(a, b), ok: i < maxIter, iterations: i };
}

/** Brent's method on a bracket (Brent 1973, ch. 4): bisection's guarantee with superlinear speed on smooth f. */
export function brent(f: (x: number) => number, a: number, b: number, opt: { xtol?: number; maxIter?: number } = {}): Root {
  let fa = f(a), fb = f(b);
  if (fa === 0) return { x: a, ok: true, iterations: 0 };
  if (fb === 0) return { x: b, ok: true, iterations: 0 };
  if (!(Math.sign(fa) * Math.sign(fb) < 0)) return { x: NaN, ok: false, iterations: 0 };
  const maxIter = opt.maxIter ?? 500;
  let c = a, fc = fa, d = b - a, e = d;
  for (let i = 1; i <= maxIter; i++) {
    if (Math.sign(fb) === Math.sign(fc)) {
      c = a;
      fc = fa;
      d = e = b - a;
    }
    if (Math.abs(fc) < Math.abs(fb)) {
      a = b; b = c; c = a;
      fa = fb; fb = fc; fc = fa;
    }
    const tol = 2 * Number.EPSILON * Math.abs(b) + 0.5 * (opt.xtol ?? 0);
    // c − b, halved; halves first only when the difference overflows (halving first would round adjacent subnormal
    // doubles apart and the bracket would never be seen as closed)
    const m = Number.isFinite(c - b) ? 0.5 * (c - b) : 0.5 * c - 0.5 * b;
    if (Math.abs(m) <= tol || fb === 0) return { x: b, ok: true, iterations: i };
    if (Math.abs(e) >= tol && Math.abs(fa) > Math.abs(fb)) {
      // inverse quadratic or secant
      let p: number, q: number;
      const s = fb / fa;
      if (a === c) {
        p = 2 * m * s;
        q = 1 - s;
      } else {
        const qa = fa / fc, r = fb / fc;
        p = s * (2 * m * qa * (qa - r) - (b - a) * (r - 1));
        q = (qa - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      else p = -p;
      // Brent's acceptance test (procedure zero): take the interpolated step only when it lands well inside the
      // bracket and is less than half the step before last; otherwise bisect
      const ePrev = e;
      e = d;
      if (2 * p < 3 * m * q - Math.abs(tol * q) && p < Math.abs(0.5 * ePrev * q)) d = p / q;
      else d = e = m;
    } else {
      d = m;
      e = m;
    }
    a = b;
    fa = fb;
    b += Math.abs(d) > tol ? d : m > 0 ? tol : -tol;
    fb = f(b);
    if (Number.isNaN(fb)) return { x: NaN, ok: false, iterations: i };
  }
  return { x: b, ok: false, iterations: maxIter };
}

/**
 * Scan [a, b] at n equal steps and refine every sign change by bisection. Can miss pairs of roots between samples and
 * roots where f touches 0 without crossing: use only when the function's wiggles are known to be wider than a step.
 */
export function scanRoots(f: (x: number) => number, a: number, b: number, n: number): number[] {
  const out: number[] = [];
  let x0 = a, f0 = f(a);
  if (f0 === 0) out.push(a);
  for (let i = 1; i <= n; i++) {
    const x1 = a + ((b - a) * i) / n, f1 = f(x1);
    if (f1 === 0) out.push(x1);
    else if (f0 !== 0 && Math.sign(f0) * Math.sign(f1) < 0) {
      const r = bisect(f, x0, x1);
      if (r.ok) out.push(r.x);
    }
    x0 = x1;
    f0 = f1;
  }
  return out;
}

/** Golden-section search for the maximum of a function that is unimodal on [a, b]. */
export function goldenMax(g: (x: number) => number, a: number, b: number, iterations = 120): { x: number; value: number } {
  const r = (Math.sqrt(5) - 1) / 2;
  let x1 = b - r * (b - a), x2 = a + r * (b - a);
  let g1 = g(x1), g2 = g(x2);
  for (let i = 0; i < iterations && x1 !== x2; i++) {
    if (g1 < g2) {
      a = x1;
      x1 = x2;
      g1 = g2;
      x2 = a + r * (b - a);
      g2 = g(x2);
    } else {
      b = x2;
      x2 = x1;
      g2 = g1;
      x1 = b - r * (b - a);
      g1 = g(x1);
    }
  }
  const x = g1 >= g2 ? x1 : x2;
  return { x, value: Math.max(g1, g2) };
}

export interface LobeCrossing {
  /** the lobe [edges[i], edges[i + 1]] the crossing lies in */
  lobe: number;
  /** where g changes sign: the rising crossing (left of the lobe's peak) and the falling one (right of it) */
  rising: number;
  falling: number;
  /** g at the lobe's peak (> 0 when the lobe crosses) */
  peak: number;
}

/**
 * Crossings of g = 0 for a function made of lobes: on each interval [edges[i], edges[i + 1]] g must rise to a single
 * peak and fall again (g ≤ 0 at both edges, unimodal inside) — e.g. g(x) = |x·sin(1/x)| − ε with edges at the zeros
 * 1/(kπ) of sin(1/x). Each lobe's peak is found by golden section; a lobe whose peak stays ≤ 0 has no crossing (and
 * is not reported); otherwise its two crossings are found by bisection on each side of the peak. Lobes are examined
 * in the order of `edges`, so with `firstOnly` the first crossing lobe in that order is the answer (the D01 largest δ:
 * edges running outward from the limit point).
 */
export function lobeCrossings(g: (x: number) => number, edges: number[], opt: { firstOnly?: boolean } = {}): LobeCrossing[] {
  const out: LobeCrossing[] = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const lo = Math.min(edges[i], edges[i + 1]), hi = Math.max(edges[i], edges[i + 1]);
    const top = goldenMax(g, lo, hi);
    if (!(top.value > 0)) continue;
    const left = bisect(g, lo, top.x), right = bisect(g, top.x, hi);
    if (!left.ok || !right.ok) continue;
    out.push({ lobe: i, rising: left.x, falling: right.x, peak: top.value });
    if (opt.firstOnly) break;
  }
  return out;
}
