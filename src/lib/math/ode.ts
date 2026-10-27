// Ordinary differential equations y′ = f(t, y) for the slope-field, vector-field and oscillation pages.
//
// - euler: the explicit Euler rule as a finite rule (D27's polyline and its h → 0), the last step landing on the end.
// - rk4: classical fixed-step Runge–Kutta (global error O(h⁴)); for drawing at a known step and for checks.
// - dopri5: Dormand–Prince 5(4) with error control (Hairer, Nørsett & Wanner, Solving ODE I, §II.4–5), the standard
//   adaptive method. When a problem shows in its steps it stops and reports why — the solution left the box the page
//   draws, |f| grew beyond a bound (a blow-up such as y′ = y² at x → 1), the step fell below hMin (a stiff or singular
//   spot), a non-finite value appeared, a barrier the page named was crossed, or the step budget ran out. The page
//   marks the stop instead of drawing on.
//   What no sampling method can promise: a singularity that a step jumps over without its effect reaching the error
//   estimate goes unnoticed (y′ = 1/(t − 1)² started at y = 1e12 steps from 0.62 to 2). Where the page knows its
//   singular lines (a pole at t = 1, y = 0 for y′ = 1/y), it passes them as `barrier`; otherwise a modest hMax.
// - expm2: the closed-form exp(tA) for a 2×2 matrix (x′ = Ax), exact up to rounding, stable when the eigenvalues
//   nearly coincide.

export type Vec = number[];
export type Field = (t: number, y: Vec) => Vec;

export function rk4Step(f: Field, t: number, y: Vec, h: number): Vec {
  const n = y.length;
  const k1 = f(t, y);
  const y2 = new Array<number>(n), y3 = new Array<number>(n), y4 = new Array<number>(n);
  for (let i = 0; i < n; i++) y2[i] = y[i] + (h / 2) * k1[i];
  const k2 = f(t + h / 2, y2);
  for (let i = 0; i < n; i++) y3[i] = y[i] + (h / 2) * k2[i];
  const k3 = f(t + h / 2, y3);
  for (let i = 0; i < n; i++) y4[i] = y[i] + h * k3[i];
  const k4 = f(t + h, y4);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return out;
}

/** Fixed-step RK4 from t0 to t1 in `steps` equal steps; returns every point (steps + 1 of them). */
export function rk4(f: Field, t0: number, y0: Vec, t1: number, steps: number): { t: number[]; y: Vec[] } {
  const h = (t1 - t0) / steps;
  const t = [t0], y = [y0.slice()];
  let cur = y0.slice();
  for (let i = 1; i <= steps; i++) {
    cur = rk4Step(f, t0 + (i - 1) * h, cur, h);
    t.push(t0 + i * h);
    y.push(cur);
  }
  return { t, y };
}

/**
 * Euler's method for a scalar y′ = f(t, y) from (t0, y0) to t1 in steps of h > 0 towards t1 (either direction): the
 * rule y_{k+1} = y_k + h_k·f(t_k, y_k) with t_k = t0 ± k·h, the last step shortened so it lands exactly on t1 (D27). A
 * finite rule: the numbers are this rule's own results in doubles, nothing is solved. Every point is returned (n + 1 of
 * them, n = ⌈|t1 − t0|/h⌉, the ceiling taken with a 1e−9 relative slack so 2/(1/128) is 256 steps, not 257); it
 * stops early ('not-finite') at the first value that is not a finite number — the rule overflowed, or f has no value
 * there — keeping the points before it. `maxSteps` (default 2²⁰) refuses a longer run with 'max-steps' and no points
 * beyond the start.
 */
export function euler(f: (t: number, y: number) => number, t0: number, y0: number, t1: number, h: number, maxSteps = 1 << 20): { t: Float64Array; y: Float64Array; stop: 'done' | 'not-finite' | 'max-steps' } {
  const span = Math.abs(t1 - t0), dir = t1 >= t0 ? 1 : -1;
  if (!(h > 0) || !Number.isFinite(span)) return { t: Float64Array.of(t0), y: Float64Array.of(y0), stop: 'not-finite' };
  const q = span / h;
  const n = span === 0 ? 0 : Math.max(1, Math.ceil(q - 1e-9 * Math.max(1, q)));
  if (n > maxSteps) return { t: Float64Array.of(t0), y: Float64Array.of(y0), stop: 'max-steps' };
  const T = new Float64Array(n + 1), Y = new Float64Array(n + 1);
  T[0] = t0;
  Y[0] = y0;
  if (!Number.isFinite(y0)) return { t: T.subarray(0, 1), y: Y.subarray(0, 1), stop: 'not-finite' };
  for (let k = 0; k < n; k++) {
    const last = k + 1 === n, tk = T[k];
    // a whole step is ±h itself (not a difference of two rounded abscissas); the last one is what is left to t1
    const yk = Y[k] + (last ? t1 - tk : dir * h) * f(tk, Y[k]);
    const next = last ? t1 : t0 + dir * (k + 1) * h;
    if (!Number.isFinite(yk)) return { t: T.subarray(0, k + 1), y: Y.subarray(0, k + 1), stop: 'not-finite' };
    T[k + 1] = next;
    Y[k + 1] = yk;
  }
  return { t: T, y: Y, stop: 'done' };
}

export type StopReason = 'done' | 'left-box' | 'blew-up' | 'step-underflow' | 'not-finite' | 'barrier' | 'max-steps';

export interface Dopri5Options {
  /** relative tolerance (default 1e-8) */
  rtol?: number;
  /** absolute tolerance (default 1e-10) */
  atol?: number;
  /** first step (default: a small fraction of the interval) */
  h0?: number;
  /** largest step, e.g. so a drawn curve has enough points (default: the whole interval) */
  hMax?: number;
  /** smallest step before giving up (default 1e-10 × the interval, at least 1e-14) */
  hMin?: number;
  /** step budget (default 100 000) */
  maxSteps?: number;
  /** keep solving only while this holds for the new state (the drawn box, usually several times the view) */
  inside?: (y: Vec) => boolean;
  /** stop when any |f| component exceeds this (default 1e6) */
  maxRate?: number;
  /**
   * A singular line the solution must not be carried across: stop (without the step) when its sign differs between
   * the two ends of a step, e.g. (t) => t − 1 for a pole at t = 1, or (t, y) => y[0] for y′ = 1/y.
   */
  barrier?: (t: number, y: Vec) => number;
}

export interface Solution {
  t: number[];
  y: Vec[];
  /** why it stopped; anything but 'done' must be shown, not connected past */
  stop: StopReason;
}

// Dormand–Prince coefficients
const C2 = 1 / 5, C3 = 3 / 10, C4 = 4 / 5, C5 = 8 / 9;
const A21 = 1 / 5;
const A31 = 3 / 40, A32 = 9 / 40;
const A41 = 44 / 45, A42 = -56 / 15, A43 = 32 / 9;
const A51 = 19372 / 6561, A52 = -25360 / 2187, A53 = 64448 / 6561, A54 = -212 / 729;
const A61 = 9017 / 3168, A62 = -355 / 33, A63 = 46732 / 5247, A64 = 49 / 176, A65 = -5103 / 18656;
const B1 = 35 / 384, B3 = 500 / 1113, B4 = 125 / 192, B5 = -2187 / 6784, B6 = 11 / 84;
// error coefficients: b (5th order) − b* (4th order)
const E1 = 71 / 57600, E3 = -71 / 16695, E4 = 71 / 1920, E5 = -17253 / 339200, E6 = 22 / 525, E7 = -1 / 40;

/** Adaptive Dormand–Prince 5(4) from t0 towards t1 (either direction). */
export function dopri5(f: Field, t0: number, y0: Vec, t1: number, opt: Dopri5Options = {}): Solution {
  const n = y0.length;
  const rtol = opt.rtol ?? 1e-8, atol = opt.atol ?? 1e-10;
  const span = Math.abs(t1 - t0), dir = t1 >= t0 ? 1 : -1;
  const hMax = opt.hMax ?? Infinity, hMin = opt.hMin ?? Math.max(1e-14, 1e-10 * span);
  const maxSteps = opt.maxSteps ?? 100000, maxRate = opt.maxRate ?? 1e6;
  const inside = opt.inside ?? (() => true);
  const barrier = opt.barrier;
  const sideOf = (t: number, y: Vec) => (barrier ? Math.sign(barrier(t, y)) : 0);
  const ts = [t0], ys = [y0.slice()];
  if (span === 0) return { t: ts, y: ys, stop: 'done' };
  let t = t0, y = y0.slice();
  let k1 = f(t, y);
  if (!k1.every(Number.isFinite)) return { t: ts, y: ys, stop: 'not-finite' };
  if (k1.some((v) => Math.abs(v) > maxRate)) return { t: ts, y: ys, stop: 'blew-up' };
  let h = Math.min(hMax, opt.h0 ?? span / 100);
  const tmp = new Array<number>(n);
  const stage = (c: number[][], w: number[]) => {
    for (let i = 0; i < n; i++) {
      let s = y[i];
      for (let j = 0; j < w.length; j++) s += dir * h * w[j] * c[j][i];
      tmp[i] = s;
    }
    return tmp.slice();
  };
  for (let step = 0; step < maxSteps; step++) {
    if (dir * (t1 - t) <= 0) return { t: ts, y: ys, stop: 'done' };
    const remaining = Math.abs(t1 - t);
    const last = h >= remaining;
    if (last) h = remaining;
    const k2 = f(t + dir * C2 * h, stage([k1], [A21]));
    const k3 = f(t + dir * C3 * h, stage([k1, k2], [A31, A32]));
    const k4 = f(t + dir * C4 * h, stage([k1, k2, k3], [A41, A42, A43]));
    const k5 = f(t + dir * C5 * h, stage([k1, k2, k3, k4], [A51, A52, A53, A54]));
    const k6 = f(t + dir * h, stage([k1, k2, k3, k4, k5], [A61, A62, A63, A64, A65]));
    const yNew = stage([k1, k3, k4, k5, k6], [B1, B3, B4, B5, B6]);
    const k7 = f(t + dir * h, yNew);
    let err = 0;
    let finite = yNew.every(Number.isFinite) && k7.every(Number.isFinite);
    for (let i = 0; i < n && finite; i++) {
      const e = dir * h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
      const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(yNew[i]));
      err += (e / sc) ** 2;
    }
    err = Math.sqrt(err / n);
    if (!finite || !(err <= 1)) {
      // reject: shrink (a non-finite trial is treated as a very bad step)
      const factor = finite ? Math.max(0.2, 0.9 * err ** -0.2) : 0.2;
      h *= factor;
      if (h < hMin) return { t: ts, y: ys, stop: finite ? 'step-underflow' : 'not-finite' };
      continue;
    }
    const tNew = last ? t1 : t + dir * h; // land exactly on t1
    if (barrier && sideOf(tNew, yNew) !== sideOf(t, y)) return { t: ts, y: ys, stop: 'barrier' };
    t = tNew;
    y = yNew;
    k1 = k7; // first-same-as-last
    if (!inside(y)) {
      ts.push(t);
      ys.push(y.slice());
      return { t: ts, y: ys, stop: 'left-box' };
    }
    ts.push(t);
    ys.push(y.slice());
    if (k7.some((v) => Math.abs(v) > maxRate)) return { t: ts, y: ys, stop: 'blew-up' };
    const grow = err === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * err ** -0.2));
    h = Math.min(hMax, h * grow);
  }
  return { t: ts, y: ys, stop: dir * (t1 - t) <= 0 ? 'done' : 'max-steps' };
}

/** A 2×2 matrix [[a, b], [c, d]] as [a, b, c, d]. */
export type M2 = [number, number, number, number];

/**
 * exp(tA) for A = [[a, b], [c, d]], in closed form. With s = (a + d)/2, h = (a − d)/2 and N = A − sI, N² = δI where
 * δ = h² + bc, so exp(tA) = e^{st}(C·I + S·N) with C = cosh(√δ t), S = sinh(√δ t)/√δ for δ > 0, their trigonometric
 * forms for δ < 0, and C = 1, S = t for δ = 0. C and S come from the series in δt² when it is small, so nearly equal
 * eigenvalues lose nothing. For δ > 0 the entries are written as e^{(s±r)t}-weighted sums with r = √δ, never as
 * cosh − sinh, which would cancel the decaying mode away (exp of diag(20, −20) lost e^{−20} entirely).
 */
export function expm2(A: M2, t: number): M2 {
  const [a, b, c, d] = A;
  const s = (a + d) / 2;
  const half = (a - d) / 2;
  const delta = half * half + b * c;
  const z = delta * t * t; // the only combination C and S/t depend on
  if (z >= 1e-4) {
    // real eigenvalues s ± r: exp(tA) = [e^{(s+r)t}(I + N/r) + e^{(s−r)t}(I − N/r)]/2
    const r = Math.sqrt(delta), w = Math.abs(r * t);
    const big = Math.exp(s * t + w), small = Math.exp(s * t - w);
    const ePlus = t >= 0 ? big : small, eMinus = t >= 0 ? small : big; // e^{(s+r)t}, e^{(s−r)t}
    // 1 ± h/r, the one that could cancel formed as bc/(r(r ± h)) (since r² − h² = bc)
    const onePlus = half >= 0 ? 1 + half / r : (b * c) / (r * (r - half));
    const oneMinus = half >= 0 ? (b * c) / (r * (r + half)) : 1 - half / r;
    // (e^{(s+r)t} − e^{(s−r)t})/(2r) = ±e^{st+w}(1 − e^{−2w})/(2r), with no difference of large numbers
    const sh = (t >= 0 ? 1 : -1) * big * (-Math.expm1(-2 * w) / (2 * r));
    // halve the weights first (exact): e^{(s+r)t}·(1 + h/r) may overflow on the way to a finite entry
    const hp = onePlus / 2, hm = oneMinus / 2;
    return [ePlus * hp + eMinus * hm, b * sh, c * sh, ePlus * hm + eMinus * hp];
  }
  let C: number, St: number; // S = t·St
  if (Math.abs(z) < 1e-4) {
    // cosh √z = Σ zᵏ/(2k)!, sinh(√z)/√z = Σ zᵏ/(2k+1)!
    C = 1 + z / 2 + (z * z) / 24 + (z * z * z) / 720;
    St = 1 + z / 6 + (z * z) / 120 + (z * z * z) / 5040;
  } else {
    const r = Math.sqrt(-z);
    C = Math.cos(r);
    St = Math.sin(r) / r;
  }
  const S = t * St;
  const e = Math.exp(s * t);
  return [e * (C + S * half), e * S * b, e * S * c, e * (C - S * half)];
}

/** Apply a 2×2 matrix to a vector. */
export const apply2 = (M: M2, v: [number, number]): [number, number] => [M[0] * v[0] + M[1] * v[1], M[2] * v[0] + M[3] * v[1]];

// ---------------------------------------------------------------------------------------------
// flow2: dopri5's method for the hot loop of a 2D autonomous field x′ = F(x) (the vector-field family: thousands of
// particles and ink points, each carried a frame at a time). The same Dormand–Prince 5(4) pair and error norm as
// dopri5, unrolled for two components, with no allocation and the point's own step size carried from one call to the
// next (out[2]), so a point in a calm region takes one step per frame and one near a stiff spot keeps its small step.
// ---------------------------------------------------------------------------------------------

/** A 2D autonomous field written into out[0], out[1] (NaN where it has no value): no allocation per call. */
export type Field2 = (x: number, y: number, out: Float64Array) => void;

/** Why flow2 stopped: 0 the whole time went by; otherwise as dopri5's StopReason. */
export const FLOW2 = { done: 0, leftBox: 1, blewUp: 2, stepUnderflow: 3, notFinite: 4, maxSteps: 5, barrier: 6 } as const;
export type Flow2Stop = (typeof FLOW2)[keyof typeof FLOW2];

export interface Flow2Options {
  /** relative tolerance (default 1e-8) */
  rtol?: number;
  /** absolute tolerance (default 1e-10) */
  atol?: number;
  /** smallest step before giving up (default max(1e-14, 1e-10 × |dt|)) */
  hMin?: number;
  /** largest step (default none: the step is cut to what is left of dt within the call, and the step carried to the
   *  next call is not shrunk by a short dt — Codex review 2026-10-07) */
  hMax?: number;
  /** step budget for this call (default 10 000) */
  maxSteps?: number;
  /** stop when |F| (either component) exceeds this (default 1e6) */
  maxRate?: number;
  /** keep going only while the new point is inside (the box the page draws, usually several times the view) */
  inside?: (x: number, y: number) => boolean;
  /** A singular line the page knows (a pole on x = 1): stop, without the step, where its sign differs between the two
   *  ends of a step — error control alone can step over a pole unseen (dopri5's `barrier`). */
  barrier?: (x: number, y: number) => number;
}

const fk = Array.from({ length: 7 }, () => new Float64Array(2));

/**
 * Carry the point (x, y) through time dt (either sign) along x′ = F(x). The end — or where it stopped — goes to
 * out[0], out[1]; out[2] is the step size: read as the step to try first (0 or less: the whole of dt), written as the
 * step to try next time; out[3], when out has it, gets the number of steps tried (for a caller's work budget), and
 * out[4] the time actually gone by (signed: less than dt where it stopped).
 * Returns the stop code (FLOW2): anything but `done` must be shown, not flowed past. A point stopped by `inside` is
 * left at its first position outside, as dopri5 does. Like dopri5 this is an explicit method: a stiff field costs
 * many small steps (the budget stops it), it is never solved implicitly.
 */
export function flow2(F: Field2, x: number, y: number, dt: number, out: Float64Array, opt: Flow2Options = {}): Flow2Stop {
  out[0] = x;
  out[1] = y;
  if (out.length > 3) out[3] = 0;
  if (out.length > 4) out[4] = 0;
  const span = Math.abs(dt);
  if (span === 0) return FLOW2.done;
  const dir = dt > 0 ? 1 : -1;
  const rtol = opt.rtol ?? 1e-8, atol = opt.atol ?? 1e-10;
  const hMax = opt.hMax ?? Infinity, hMin = opt.hMin ?? Math.max(1e-14, 1e-10 * span);
  const maxSteps = opt.maxSteps ?? 10000, maxRate = opt.maxRate ?? 1e6;
  const inside = opt.inside, barrier = opt.barrier;
  const [k1, k2, k3, k4, k5, k6, k7] = fk;
  let h = out[2] > 0 ? Math.min(out[2], hMax) : Math.min(span, hMax);
  let done = 0; // time gone by (≥ 0)
  let tried = 0;
  const count = out.length > 3;
  F(x, y, k1);
  if (!Number.isFinite(k1[0]) || !Number.isFinite(k1[1])) { out[2] = h; return FLOW2.notFinite; }
  if (Math.abs(k1[0]) > maxRate || Math.abs(k1[1]) > maxRate) { out[2] = h; return FLOW2.blewUp; }
  for (let step = 0; step < maxSteps; step++) {
    const remaining = span - done;
    if (remaining <= 0) break;
    tried++;
    if (count) out[3] = tried;
    const last = h >= remaining;
    const hs = last ? remaining : h;
    const s = dir * hs;
    F(x + s * A21 * k1[0], y + s * A21 * k1[1], k2);
    F(x + s * (A31 * k1[0] + A32 * k2[0]), y + s * (A31 * k1[1] + A32 * k2[1]), k3);
    F(x + s * (A41 * k1[0] + A42 * k2[0] + A43 * k3[0]), y + s * (A41 * k1[1] + A42 * k2[1] + A43 * k3[1]), k4);
    F(x + s * (A51 * k1[0] + A52 * k2[0] + A53 * k3[0] + A54 * k4[0]), y + s * (A51 * k1[1] + A52 * k2[1] + A53 * k3[1] + A54 * k4[1]), k5);
    F(x + s * (A61 * k1[0] + A62 * k2[0] + A63 * k3[0] + A64 * k4[0] + A65 * k5[0]), y + s * (A61 * k1[1] + A62 * k2[1] + A63 * k3[1] + A64 * k4[1] + A65 * k5[1]), k6);
    const xn = x + s * (B1 * k1[0] + B3 * k3[0] + B4 * k4[0] + B5 * k5[0] + B6 * k6[0]);
    const yn = y + s * (B1 * k1[1] + B3 * k3[1] + B4 * k4[1] + B5 * k5[1] + B6 * k6[1]);
    F(xn, yn, k7);
    const finite = Number.isFinite(xn) && Number.isFinite(yn) && Number.isFinite(k7[0]) && Number.isFinite(k7[1]);
    let err = Infinity;
    if (finite) {
      const ex = s * (E1 * k1[0] + E3 * k3[0] + E4 * k4[0] + E5 * k5[0] + E6 * k6[0] + E7 * k7[0]);
      const ey = s * (E1 * k1[1] + E3 * k3[1] + E4 * k4[1] + E5 * k5[1] + E6 * k6[1] + E7 * k7[1]);
      const sx = atol + rtol * Math.max(Math.abs(x), Math.abs(xn)), sy = atol + rtol * Math.max(Math.abs(y), Math.abs(yn));
      err = Math.sqrt(((ex / sx) ** 2 + (ey / sy) ** 2) / 2);
    }
    if (!(err <= 1)) {
      // reject: shrink (a non-finite trial is a very bad step)
      h = hs * (finite ? Math.max(0.2, 0.9 * err ** -0.2) : 0.2);
      if (h < hMin) { out[2] = hMin; return finite ? FLOW2.stepUnderflow : FLOW2.notFinite; }
      continue;
    }
    if (barrier && Math.sign(barrier(xn, yn)) !== Math.sign(barrier(x, y))) { out[2] = h; return FLOW2.barrier; }
    x = xn;
    y = yn;
    done = last ? span : done + hs;
    out[0] = x;
    out[1] = y;
    if (out.length > 4) out[4] = dir * done;
    k1[0] = k7[0]; // first-same-as-last
    k1[1] = k7[1];
    const grow = err === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * err ** -0.2));
    // the step to try next: grown from the one just taken, but a short last piece of dt does not shrink it
    h = Math.min(hMax, Math.max(last ? h : 0, hs * grow));
    if (inside && !inside(x, y)) { out[2] = h; return FLOW2.leftBox; }
    if (Math.abs(k7[0]) > maxRate || Math.abs(k7[1]) > maxRate) { out[2] = h; return FLOW2.blewUp; }
  }
  out[2] = h;
  return done >= span ? FLOW2.done : FLOW2.maxSteps;
}
