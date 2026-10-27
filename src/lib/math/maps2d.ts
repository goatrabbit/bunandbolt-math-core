// Iterated maps of the plane (WA04 strange attractor; design/pages/WA04-strange-attractor.md §8). Pure, double
// precision, deterministic: nothing here draws or uses randomness.
//
// What each result is (the four kinds of statement):
//   exact      `constantDet` (Hénon and Lozi −b, Ikeda u², proved in the comments), `fixedPoints` of Hénon and Lozi in
//              closed form, `kaplanYorke`'s case split (a definition, applied to whatever exponents it is given).
//   numerical  `lyapunov` (tangent-vector renormalisation over a finite orbit from one start), `period` (an orbit
//              returning within a tolerance), `classify` (both, with an "undetermined" verdict when neither decides).
//              They describe the orbit from the given start over the given steps, never the map at those parameters as
//              a whole: an escaping orbit does not mean there is no attractor, and a finite-time negative exponent is
//              not a detected period (Codex review 2026-10-10).
//
// The maps (p is the parameter list):
//   henon     (x, y) → (1 − a x² + y, b x)                          p = [a, b]            Hénon 1976
//   lozi      (x, y) → (1 − a |x| + y, b x)                         p = [a, b]            Lozi 1978
//   ikeda     z → 1 + u z e^{i t},  t = k − 6 / (1 + |z|²)           p = [u, k]            Ikeda 1979; u = 0.9, k = 0.4 classic
//   clifford  (x, y) → (sin a y + c cos a x, sin b x + d cos b y)    p = [a, b, c, d]      a popular-art family

export type MapId = 'henon' | 'lozi' | 'ikeda' | 'clifford';
export const MAP_IDS: readonly MapId[] = ['henon', 'lozi', 'ikeda', 'clifford'];

export interface Pt { x: number; y: number }
/** Orbits farther than this (|x| + |y|) count as escaped. Every attractor of these maps at the page's parameters lies
 *  within |x| + |y| < 10; Ikeda's whole plane is pulled inside |z| ≤ 1/(1 − u) for u < 1. */
export const ESCAPE = 1e6;

/** One step in place: the hot loops (density, release) reuse one object. */
export function stepper(id: MapId, p: readonly number[]): (s: Pt) => void {
  switch (id) {
    case 'henon': { const [a, b] = p; return (s) => { const x = s.x; s.x = 1 - a * x * x + s.y; s.y = b * x; }; }
    case 'lozi': { const [a, b] = p; return (s) => { const x = s.x; s.x = 1 - a * Math.abs(x) + s.y; s.y = b * x; }; }
    case 'ikeda': {
      const [u, k] = p;
      return (s) => {
        const x = s.x, y = s.y, t = k - 6 / (1 + x * x + y * y), c = Math.cos(t), sn = Math.sin(t);
        s.x = 1 + u * (x * c - y * sn); s.y = u * (x * sn + y * c);
      };
    }
    case 'clifford': {
      const [a, b, c, d] = p;
      return (s) => { const x = s.x, y = s.y; s.x = Math.sin(a * y) + c * Math.cos(a * x); s.y = Math.sin(b * x) + d * Math.cos(b * y); };
    }
  }
}

export function step(id: MapId, p: readonly number[], x: number, y: number): [number, number] {
  const s = { x, y };
  stepper(id, p)(s);
  return [s.x, s.y];
}

/** The Jacobian [[∂x′/∂x, ∂x′/∂y], [∂y′/∂x, ∂y′/∂y]] at (x, y), row by row. Lozi's at x = 0 uses sign(0) = 0 (a line
 *  the orbit meets with probability zero). */
export function jacobian(id: MapId, p: readonly number[], x: number, y: number): [number, number, number, number] {
  switch (id) {
    case 'henon': return [-2 * p[0] * x, 1, p[1], 0];
    case 'lozi': return [-p[0] * Math.sign(x), 1, p[1], 0];
    case 'ikeda': {
      const [u, k] = p, r = 1 + x * x + y * y, t = k - 6 / r, c = Math.cos(t), s = Math.sin(t);
      const tx = (12 * x) / (r * r), ty = (12 * y) / (r * r), A = x * s + y * c, B = x * c - y * s;
      return [u * (c - A * tx), u * (-s - A * ty), u * (s + B * tx), u * (c + B * ty)];
    }
    case 'clifford': {
      const [a, b, c, d] = p;
      return [-c * a * Math.sin(a * x), a * Math.cos(a * y), b * Math.cos(b * x), -d * b * Math.sin(b * y)];
    }
  }
}

/**
 * The determinant of the Jacobian when it is the same everywhere (exact), else null.
 *   Hénon, Lozi: det [[·, 1], [b, 0]] = −b.
 *   Ikeda: with A = x sin t + y cos t, B = x cos t − y sin t and t_x = 12x/r², t_y = 12y/r² (r = 1 + |z|²),
 *     det / u² = 1 + t_y (B cos t + A sin t) + t_x (B sin t − A cos t) = 1 + t_y·x − t_x·y = 1, so det = u².
 *     (Geometrically: the twist z → z e^{i t(|z|)} keeps every circle and shears along it, which keeps area.)
 *   Clifford: depends on the point.
 * Every region's area is multiplied by |det| at each step: the attractor of a map with |det| < 1 has area zero.
 */
export function constantDet(id: MapId, p: readonly number[]): number | null {
  switch (id) {
    case 'henon': case 'lozi': return -p[1];
    case 'ikeda': return p[0] * p[0];
    case 'clifford': return null;
  }
}

/** A fixed point with its Jacobian's eigenvalues (real ones only; a complex pair gives null). */
export interface FixedPoint { x: number; y: number; eig: [number, number] | null }

/** Real eigenvalues of a 2×2 matrix (row by row), larger first by value; null for a complex pair. */
export function eigen2([a, b, c, d]: readonly number[]): [number, number] | null {
  const tr = a + d, det = a * d - b * c, disc = tr * tr - 4 * det;
  if (disc < 0) return null;
  const s = Math.sqrt(disc);
  // the root without cancellation, the other from the product
  const big = tr >= 0 ? (tr + s) / 2 : (tr - s) / 2;
  const small = big !== 0 ? det / big : (tr >= 0 ? (tr - s) / 2 : (tr + s) / 2);
  return big >= small ? [big, small] : [small, big];
}

/**
 * Fixed points in closed form (exact up to the rounding of the formula), for Hénon and Lozi; [] for the others.
 *   Hénon: x = 1 − a x² + b x, i.e. a x² + (1 − b) x − 1 = 0, y = b x.
 *   Lozi: x = 1 − a|x| + b x: on x > 0, x = 1/(1 + a − b); on x < 0, x = 1/(1 − a − b); each kept only on its side.
 */
export function fixedPoints(id: MapId, p: readonly number[]): FixedPoint[] {
  const out: FixedPoint[] = [];
  const add = (x: number) => { const y = p[1] * x; out.push({ x, y, eig: eigen2(jacobian(id, p, x, y)) }); };
  if (id === 'henon') {
    const [a, b] = p;
    if (a === 0) { if (b !== 1) add(1 / (1 - b)); return out; }
    const disc = (1 - b) ** 2 + 4 * a;
    if (disc < 0) return out;
    const s = Math.sqrt(disc);
    // the root without cancellation, then the other from the product −1/a
    const q = -0.5 * ((1 - b) + Math.sign(1 - b || 1) * s);
    const r1 = q / a, r2 = q !== 0 ? -1 / q : NaN;
    for (const x of [r1, r2].filter(Number.isFinite).sort((m, n) => n - m)) add(x);
    return out;
  }
  if (id === 'lozi') {
    const [a, b] = p;
    const xp = 1 / (1 + a - b), xn = 1 / (1 - a - b);
    if (Number.isFinite(xp) && xp > 0) add(xp);
    if (Number.isFinite(xn) && xn < 0) add(xn);
  }
  return out;
}

const escaped = (s: Pt) => !(Math.abs(s.x) + Math.abs(s.y) < ESCAPE);

export interface Lyapunov {
  /** the larger exponent */
  l1: number;
  /** the smaller one: the mean of log|det J| minus l1 (−∞ when the map squashes the plane flat, b = 0) */
  l2: number;
  /** the orbit left |x| + |y| < ESCAPE (the exponents are then NaN) */
  escaped: boolean;
  /** where the orbit ended (to continue from) */
  end: Pt;
}

/**
 * Both Lyapunov exponents of the orbit from `start`: `transient` steps dropped, then the mean growth of a tangent
 * vector renormalised every step over `n` steps (l1), and the mean log|det J| minus l1 (l2). Numerical: a finite orbit
 * from one start. For a constant Jacobian l1 + l2 = ln|det| holds to rounding by construction, so it is no
 * independent check of l1.
 */
export function lyapunov(id: MapId, p: readonly number[], o: { start?: Pt; transient?: number; n?: number } = {}): Lyapunov {
  const st = stepper(id, p), s = { ...(o.start ?? { x: 0.1, y: 0.1 }) };
  const T = o.transient ?? 10_000, n = o.n ?? 1_000_000;
  const fail = (): Lyapunov => ({ l1: NaN, l2: NaN, escaped: true, end: { ...s } });
  for (let i = 0; i < T; i++) { st(s); if (escaped(s)) return fail(); }
  let vx = 1, vy = 0, s1 = 0, sd = 0;
  for (let i = 0; i < n; i++) {
    const [j11, j12, j21, j22] = jacobian(id, p, s.x, s.y);
    const wx = j11 * vx + j12 * vy, wy = j21 * vx + j22 * vy, len = Math.hypot(wx, wy);
    if (len === 0) { s1 = -Infinity; vx = 1; vy = 0; } else { s1 += Math.log(len); vx = wx / len; vy = wy / len; }
    sd += Math.log(Math.abs(j11 * j22 - j12 * j21));
    st(s);
    if (escaped(s)) return fail();
  }
  const l1 = s1 / n;
  // a tangent vector sent to 0 (a superstable point of a map that squashes the plane, det J = 0 there) makes both
  // sums −∞; the smaller exponent is then −∞ too, not −∞ − (−∞) (Codex review: Hénon (1, 0), Ikeda (0, 0.4))
  const l2 = l1 === -Infinity ? -Infinity : sd / n - l1;
  return { l1, l2, escaped: false, end: { ...s } };
}

/**
 * The smallest p ≤ maxP with |f^p(z) − z| < tol (max norm) after `transient` steps, 0 when none, −1 when the orbit
 * escaped. Numerical: an orbit that comes back within tol, not a proof of an attracting cycle.
 */
export function period(id: MapId, p: readonly number[], o: { start?: Pt; transient?: number; maxP?: number; tol?: number } = {}): number {
  const st = stepper(id, p), s = { ...(o.start ?? { x: 0.1, y: 0.1 }) };
  const T = o.transient ?? 100_000, maxP = o.maxP ?? 512, tol = o.tol ?? 1e-9;
  for (let i = 0; i < T; i++) { st(s); if (escaped(s)) return -1; }
  const x0 = s.x, y0 = s.y;
  for (let k = 1; k <= maxP; k++) {
    st(s);
    if (escaped(s)) return -1;
    if (Math.max(Math.abs(s.x - x0), Math.abs(s.y - y0)) < tol) return k;
  }
  return 0;
}

export type Verdict =
  | { kind: 'escape' }
  | { kind: 'periodic'; period: number; l1: number; l2: number }
  | { kind: 'chaotic'; l1: number; l2: number }
  | { kind: 'undetermined'; l1: number; l2: number };

/** λ₁ must clear this to call an orbit chaotic (the page shows three decimals; starts differ in the fourth). */
export const CHAOS_MIN = 0.01;

/**
 * What the orbit from `start` does (numerical, that orbit only): escapes; comes back to itself within tol (periodic);
 * has no period up to maxP and λ₁ > CHAOS_MIN (chaotic); anything else is undetermined (λ₁ near 0, or negative without
 * a detected period: a long cycle, or a slow approach).
 */
export function classify(id: MapId, p: readonly number[], o: { start?: Pt; transient?: number; n?: number; maxP?: number; tol?: number } = {}): Verdict {
  const per = period(id, p, { start: o.start, transient: o.transient ?? 100_000, maxP: o.maxP, tol: o.tol });
  if (per < 0) return { kind: 'escape' };
  const L = lyapunov(id, p, { start: o.start, transient: o.transient ?? 100_000, n: o.n ?? 200_000 });
  if (L.escaped) return { kind: 'escape' };
  if (per > 0) return { kind: 'periodic', period: per, l1: L.l1, l2: L.l2 };
  if (L.l1 > CHAOS_MIN) return { kind: 'chaotic', l1: L.l1, l2: L.l2 };
  return { kind: 'undetermined', l1: L.l1, l2: L.l2 };
}

/**
 * The Kaplan–Yorke (Lyapunov) dimension in the plane, by its definition with j the largest count of exponents whose
 * sum is ≥ 0:  l1 < 0 → 0 (j = 0);  l1 ≥ 0 > l1 + l2 → 1 + l1/|l2|;  l1 + l2 ≥ 0 → 2 (j = 2, the plane's dimension).
 * A number computed from two exponents: it is not the box-counting dimension, and 2 does not prove positive area;
 * 0 says the orbit contracts in every direction, not that a cycle was found.
 */
export function kaplanYorke(l1: number, l2: number): { value: number; branch: 'contracting' | 'fractional' | 'full' } | null {
  if (Number.isNaN(l1) || Number.isNaN(l2) || l1 === Infinity) return null;
  if (l1 < 0) return { value: 0, branch: 'contracting' };
  if (l1 + l2 >= 0) return { value: 2, branch: 'full' };
  return { value: l2 === -Infinity ? 1 : 1 + l1 / Math.abs(l2), branch: 'fractional' };
}
