// Numerical integration with an error estimate. Use a closed form when the page has one (most presets do); these are
// for what the reader types or draws, and for checks.
//
// - adaptiveSimpson: recursive Simpson with Richardson correction; stops a branch when |S₂ − S| < 15·tol (Lyness), so
//   the estimate is reliable for smooth integrands that its samples resolve. It reports `ok: false` when the depth
//   budget runs out (a kink, a singularity, an oscillation too fast to resolve), with the best value so far. Every
//   branch is split at least `minDepth` times (default 4: 65 samples) before it may stop, so a wave that happens to
//   be 1 at all five first samples (cos 8πx on [0, 1]) is not taken for a constant — but no sampling scheme can rule
//   out aliasing for every frequency: integrate fast oscillations panel by panel between known zeros.
// - gaussLegendre: n-point rule on [a, b] (nodes by Newton on Pₙ, exact for polynomials of degree 2n − 1);
//   gaussLegendreEst compares n with 2n points, composite over panels for long or wiggly ranges.
// - diskIntegral, rectIntegral: ∬ over a disc (polar: Gauss–Legendre in the radius, trapezoid in the angle) or a
//   rectangle (Gauss–Legendre both ways), each direction refined on its own.

export interface QuadResult {
  value: number;
  /** estimated absolute error */
  error: number;
  /** false when the method could not meet the tolerance (value is the best available) */
  ok: boolean;
  evaluations: number;
}

export function adaptiveSimpson(f: (x: number) => number, a: number, b: number, opt: { tol?: number; maxDepth?: number; minDepth?: number } = {}): QuadResult {
  const tol = opt.tol ?? 1e-10, maxDepth = opt.maxDepth ?? 50, minDepth = Math.min(opt.minDepth ?? 4, maxDepth);
  let evaluations = 0, ok = true, error = 0;
  const F = (x: number) => {
    evaluations++;
    return f(x);
  };
  const simpson = (fa: number, fm: number, fb: number, w: number) => (w / 6) * (fa + 4 * fm + fb);
  function rec(a: number, b: number, fa: number, fm: number, fb: number, whole: number, tol: number, depth: number): number {
    const m = (a + b) / 2, lm = (a + m) / 2, rm = (m + b) / 2;
    const flm = F(lm), frm = F(rm);
    const left = simpson(fa, flm, fm, m - a), right = simpson(fm, frm, fb, b - m);
    const diff = left + right - whole;
    if (!Number.isFinite(diff)) {
      ok = false;
      return left + right;
    }
    // converged (after the minimum splits) is checked before the budget, so a branch that converges on its last
    // allowed level counts as converged
    if (maxDepth - depth >= minDepth && Math.abs(diff) <= 15 * tol) {
      error += Math.abs(diff) / 15;
      return left + right + diff / 15;
    }
    if (depth <= 0) {
      ok = false;
      error += Math.abs(diff) / 15;
      return left + right + diff / 15;
    }
    return rec(a, m, fa, flm, fm, left, tol / 2, depth - 1) + rec(m, b, fm, frm, fb, right, tol / 2, depth - 1);
  }
  if (a === b) return { value: 0, error: 0, ok: true, evaluations: 0 };
  const fa = F(a), fb = F(b), fm = F((a + b) / 2);
  const value = rec(a, b, fa, fm, fb, simpson(fa, fm, fb, b - a), tol, maxDepth);
  return { value, error, ok: ok && Number.isFinite(value), evaluations };
}

const glCache = new Map<number, { x: number[]; w: number[] }>();

/** Nodes and weights of the n-point Gauss–Legendre rule on [−1, 1] (n ≥ 1). */
export function gaussLegendreNodes(n: number): { x: number[]; w: number[] } {
  const hit = glCache.get(n);
  if (hit) return hit;
  const x = new Array<number>(n), w = new Array<number>(n);
  const m = Math.floor((n + 1) / 2);
  for (let i = 0; i < m; i++) {
    // Newton from the Tricomi-style initial guess
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      dp = (n * (z * p1 - p2)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    // recompute the derivative at the converged node for the weight
    let p1 = 1, p2 = 0;
    for (let j = 1; j <= n; j++) {
      const p3 = p2;
      p2 = p1;
      p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
    }
    dp = (n * (z * p1 - p2)) / (z * z - 1);
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  const out = { x, w };
  glCache.set(n, out);
  return out;
}

/** n-point Gauss–Legendre on [a, b], optionally over `panels` equal panels. */
export function gaussLegendre(f: (x: number) => number, a: number, b: number, n = 10, panels = 1): number {
  const { x, w } = gaussLegendreNodes(n);
  const width = (b - a) / panels;
  let sum = 0;
  for (let p = 0; p < panels; p++) {
    const lo = a + p * width, half = width / 2, mid = lo + half;
    let s = 0;
    for (let i = 0; i < n; i++) s += w[i] * f(mid + half * x[i]);
    sum += half * s;
  }
  return sum;
}

/** Gauss–Legendre with an error estimate: n against 2n points (same panels). */
export function gaussLegendreEst(f: (x: number) => number, a: number, b: number, n = 10, panels = 1, tol = 1e-10): QuadResult {
  const coarse = gaussLegendre(f, a, b, n, panels);
  const fine = gaussLegendre(f, a, b, 2 * n, panels);
  const error = Math.abs(fine - coarse);
  return { value: fine, error, ok: Number.isFinite(fine) && error <= Math.max(tol, tol * Math.abs(fine)), evaluations: 3 * n * panels };
}

/** A 2D product rule refined in each direction on its own: (a, b) → (2a, b) and (a, 2b) are compared with (a, b);
 *  a direction whose change is within the tolerance stops doubling, and the whole stops once two successive rounds
 *  were both within it in both directions (one agreement can be luck). `scale` is the integral of |f| by the same
 *  rule (the relative tolerance is against it). */
function settle2(rule: (a: number, b: number) => { v: number; s: number }, a0: number, b0: number, aMax: number, bMax: number, rtol: number, atol: number): QuadResult {
  let a = a0, b = b0, calm = 0, evaluations = 0, error = Infinity;
  let base = rule(a, b);
  evaluations += a * b;
  for (;;) {
    if (!Number.isFinite(base.v)) return { value: NaN, error: NaN, ok: false, evaluations };
    const ra = rule(2 * a, b), rb = rule(a, 2 * b);
    evaluations += 4 * a * b;
    // a value missing only at the finer nodes is missing (NaN would compare as within nothing and slip past the limits)
    if (!Number.isFinite(ra.v) || !Number.isFinite(rb.v)) return { value: NaN, error: NaN, ok: false, evaluations };
    const tol = rtol * base.s + atol;
    const da = Math.abs(ra.v - base.v), db = Math.abs(rb.v - base.v);
    error = da + db;
    calm = da <= tol && db <= tol ? calm + 1 : 0;
    if (calm >= 2) return { value: base.v, error, ok: Number.isFinite(base.v), evaluations };
    // refine where it still changes (both, once they agree: the second round must be at a finer rule too)
    const ga = da > tol || calm > 0, gb = db > tol || calm > 0;
    if ((ga && 2 * a > aMax) || (gb && 2 * b > bMax) || (!ga && !gb)) return { value: base.v, error, ok: false, evaluations };
    if (ga && gb) { a *= 2; b *= 2; base = rule(a, b); evaluations += a * b; }
    else if (ga) { a *= 2; base = ra; }
    else { b *= 2; base = rb; }
  }
}

/**
 * ∬ f dA over the disc (cx, cy, r): polar coordinates, Gauss–Legendre in the radius (with the weight ρ) and the
 * trapezoid rule in the angle (spectrally accurate for the periodic angle), each direction doubled on its own until two
 * successive rounds change the value by at most rtol × ∬|f| + atol (default 1e−9, 1e−14); `ok: false` when it does not
 * settle within 128 radial and 1024 angular nodes. Nodes never sit on the centre or the rim, so a singular point exactly
 * there is not sampled; a pole inside shows as a failure to settle or a wrong value, never as an error. E25 compares it
 * with the flux through the rim.
 */
export function diskIntegral(f: (x: number, y: number) => number, cx: number, cy: number, r: number, o: { rtol?: number; atol?: number } = {}): QuadResult {
  const rule = (m: number, k: number) => {
    const { x, w } = gaussLegendreNodes(m);
    let v = 0, s = 0;
    for (let i = 0; i < m; i++) {
      const rho = (r * (x[i] + 1)) / 2;
      let ring = 0, ringAbs = 0;
      for (let j = 0; j < k; j++) {
        const a = (2 * Math.PI * (j + 0.5)) / k;
        const fv = f(cx + rho * Math.cos(a), cy + rho * Math.sin(a));
        ring += fv;
        ringAbs += Math.abs(fv);
      }
      const wgt = (w[i] * rho * (2 * Math.PI) / k) * (r / 2);
      v += wgt * ring;
      s += wgt * ringAbs;
    }
    return { v, s };
  };
  return settle2(rule, 8, 32, 128, 1024, o.rtol ?? 1e-9, o.atol ?? 1e-14);
}

/** ∬ f dA over the rectangle [x0, x1] × [y0, y1]: 8-point Gauss–Legendre on equal panels in each direction, the panels
 *  doubled in each direction on its own as for diskIntegral (at most 64 panels a side). */
export function rectIntegral(f: (x: number, y: number) => number, x0: number, x1: number, y0: number, y1: number, o: { rtol?: number; atol?: number } = {}): QuadResult {
  const { x: g, w } = gaussLegendreNodes(8);
  const rule = (px: number, py: number) => {
    const hx = (x1 - x0) / px, hy = (y1 - y0) / py;
    let v = 0, s = 0;
    for (let i = 0; i < px; i++) for (let a = 0; a < 8; a++) {
      const X = x0 + hx * (i + (g[a] + 1) / 2);
      for (let j = 0; j < py; j++) for (let b = 0; b < 8; b++) {
        const fv = f(X, y0 + hy * (j + (g[b] + 1) / 2));
        const wgt = (w[a] * w[b] * hx * hy) / 4;
        v += wgt * fv;
        s += wgt * Math.abs(fv);
      }
    }
    return { v, s };
  };
  return settle2((a, b) => rule(a, b), 1, 1, 64, 64, o.rtol ?? 1e-9, o.atol ?? 1e-14);
}
