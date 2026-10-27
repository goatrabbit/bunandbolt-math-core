// A solution of y′ = f(t, y) you can ask for at any time: Dormand–Prince 5(4) with error control (as ode.ts's dopri5)
// plus its continuous extension, the fourth-order dense output of Hairer, Nørsett & Wanner (Solving ODE I, §II.6; the
// coefficients of their DOPRI5 code). The oscillation pages (G04's pendulum, G08's populations) play a numeric motion
// by the clock: the record wants the state at exact sample times and at a crossing, not at the solver's own steps.
//
// - `at(t)` integrates forward as far as needed and interpolates inside the accepted step that holds t: the interpolant
//   is fourth order, so between steps it is as good as the steps themselves (to the tolerance, not to the last bit).
// - `crossings(g, ta, tb)` finds where g(t, y(t)) changes sign: g is sampled at every step end and at a few points
//   inside each step, and each bracketed sign change is located by Brent's method on the dense output. Two sign
//   changes closer than one sample spacing cancel and are missed — keep `hMax` below the motion's fastest half swing.
// - A run that stops (a blow-up, a step that underflows, the step budget) keeps what it has: `stop` says why and `at`
//   beyond it gives NaN; the page shows the stop, it does not connect past it (as dopri5).
// Everything here is numeric (spec 2.2's four kinds of statement): the page says so where it shows the result.
import type { Field, StopReason, Vec } from './ode';
import { brent } from './roots';

// Dormand–Prince coefficients (as in ode.ts)
const C2 = 1 / 5, C3 = 3 / 10, C4 = 4 / 5, C5 = 8 / 9;
const A21 = 1 / 5;
const A31 = 3 / 40, A32 = 9 / 40;
const A41 = 44 / 45, A42 = -56 / 15, A43 = 32 / 9;
const A51 = 19372 / 6561, A52 = -25360 / 2187, A53 = 64448 / 6561, A54 = -212 / 729;
const A61 = 9017 / 3168, A62 = -355 / 33, A63 = 46732 / 5247, A64 = 49 / 176, A65 = -5103 / 18656;
const B1 = 35 / 384, B3 = 500 / 1113, B4 = 125 / 192, B5 = -2187 / 6784, B6 = 11 / 84;
const E1 = 71 / 57600, E3 = -71 / 16695, E4 = 71 / 1920, E5 = -17253 / 339200, E6 = 22 / 525, E7 = -1 / 40;
// dense output (DOPRI5's d coefficients)
const D1 = -12715105075 / 11282082432, D3 = 87487479700 / 32700410799, D4 = -10690763975 / 1880347072;
const D5 = 701980252875 / 199316789632, D6 = -1453857185 / 822651844, D7 = 69997945 / 29380423;

export interface TrajectoryOptions {
  /** relative tolerance (default 1e-10) */
  rtol?: number;
  /** absolute tolerance (default 1e-12) */
  atol?: number;
  /** largest step (default 0.1): also the scale on which crossings are looked for */
  hMax?: number;
  /** first step (default hMax / 10) */
  h0?: number;
  /** smallest step before giving up (default 1e-12) */
  hMin?: number;
  /** stop when any |f| component exceeds this (default 1e6) */
  maxRate?: number;
  /** step budget over the trajectory's whole life (default 2 000 000) */
  maxSteps?: number;
}

export interface Crossing {
  t: number;
  /** +1: g went from negative to non-negative; −1: the other way */
  dir: 1 | -1;
}

export interface Trajectory {
  readonly t0: number;
  /** how far it has been integrated */
  readonly tEnd: number;
  /** why it cannot go on (null while it can) */
  readonly stop: StopReason | null;
  /** accepted steps so far */
  readonly steps: number;
  /** integrate forward to at least t; false when it stopped before t */
  extend(t: number): boolean;
  /** the state at t ≥ t0 (integrating as needed); NaN components beyond a stop or before the steps kept */
  at(t: number): Vec;
  /** the sign changes of g(t, y(t)) in (ta, tb], increasing; `samples` points inside each step (default 4) */
  crossings(g: (t: number, y: Vec) => number, ta: number, tb: number, o?: { samples?: number; xtol?: number }): Crossing[];
  /** ∫ fn(t, y(t)) dt over [ta, tb] on the dense output: a 5-point Gauss–Legendre rule on each solver step (split in
   *  `pieces`, default 1). Exact for polynomials of degree 9 in t; accurate when fn(t, y(t)) varies slowly over a step —
   *  it has no error control of its own (Codex review 9), so a page checks it (G08: a cycle's average against the
   *  equilibrium) or passes more pieces / a smaller hMax. NaN where the run stopped before tb. */
  integrate(fn: (t: number, y: Vec) => number, ta: number, tb: number, pieces?: number): number;
  /** drop the steps that end before t (a long run keeps only the window it still draws) */
  forget(t: number): void;
}

// 5-point Gauss–Legendre on [−1, 1]
const GL_X = [0, -0.5384693101056831, 0.5384693101056831, -0.906179845938664, 0.906179845938664];
const GL_W = [0.5688888888888889, 0.47862867049936647, 0.47862867049936647, 0.23692688505618908, 0.23692688505618908];

/** Start a trajectory of y′ = f(t, y) at (t0, y0); it integrates lazily, forward in time only. */
export function createTrajectory(f: Field, t0: number, y0: Vec, opt: TrajectoryOptions = {}): Trajectory {
  const n = y0.length;
  const rtol = opt.rtol ?? 1e-10, atol = opt.atol ?? 1e-12;
  const hMax = opt.hMax ?? 0.1, hMin = opt.hMin ?? 1e-12, maxRate = opt.maxRate ?? 1e6, maxSteps = opt.maxSteps ?? 2_000_000;
  // accepted steps: start time, length, and the five interpolation vectors (n each) in one flat array per step
  const tOld: number[] = [], hs: number[] = [], cont: Float64Array[] = [];
  let first = 0; // steps before this index were forgotten
  let tMin = t0, forgotten = false; // the earliest time still answered (Codex review 7: compaction must not revive t0)
  let t = t0, y = y0.slice(), h = opt.h0 ?? hMax / 10, steps = 0;
  let stop: StopReason | null = null;
  let k1 = f(t, y);
  if (!k1.every(Number.isFinite)) stop = 'not-finite';
  else if (k1.some((v) => Math.abs(v) > maxRate)) stop = 'blew-up';
  const tmp = new Array<number>(n);
  const stage = (ks: Vec[], ws: number[]) => {
    for (let i = 0; i < n; i++) {
      let s = y[i];
      for (let j = 0; j < ws.length; j++) s += h * ws[j] * ks[j][i];
      tmp[i] = s;
    }
    return tmp.slice();
  };

  /** One accepted step (retrying smaller ones as needed); false when it stopped. */
  function step(): boolean {
    if (stop) return false;
    for (;;) {
      if (steps >= maxSteps) { stop = 'max-steps'; return false; }
      const k2 = f(t + C2 * h, stage([k1], [A21]));
      const k3 = f(t + C3 * h, stage([k1, k2], [A31, A32]));
      const k4 = f(t + C4 * h, stage([k1, k2, k3], [A41, A42, A43]));
      const k5 = f(t + C5 * h, stage([k1, k2, k3, k4], [A51, A52, A53, A54]));
      const k6 = f(t + h, stage([k1, k2, k3, k4, k5], [A61, A62, A63, A64, A65]));
      const yNew = stage([k1, k3, k4, k5, k6], [B1, B3, B4, B5, B6]);
      const k7 = f(t + h, yNew);
      const finite = yNew.every(Number.isFinite) && k7.every(Number.isFinite);
      let err = 0;
      for (let i = 0; i < n && finite; i++) {
        const e = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(yNew[i]));
        err += (e / sc) ** 2;
      }
      err = Math.sqrt(err / n);
      if (!finite || !(err <= 1)) {
        h *= finite ? Math.max(0.2, 0.9 * err ** -0.2) : 0.2;
        if (h < hMin) { stop = finite ? 'step-underflow' : 'not-finite'; return false; }
        continue;
      }
      // accepted: keep the interpolant y(t + θh) = r1 + θ(r2 + (1 − θ)(r3 + θ(r4 + (1 − θ) r5)))
      const c = new Float64Array(5 * n);
      for (let i = 0; i < n; i++) {
        const dy = yNew[i] - y[i], bspl = h * k1[i] - dy;
        c[i] = y[i];
        c[n + i] = dy;
        c[2 * n + i] = bspl;
        c[3 * n + i] = dy - h * k7[i] - bspl;
        c[4 * n + i] = h * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k7[i]);
      }
      tOld.push(t);
      hs.push(h);
      cont.push(c);
      steps++;
      t += h;
      y = yNew;
      k1 = k7;
      if (k7.some((v) => Math.abs(v) > maxRate)) { stop = 'blew-up'; return false; }
      const grow = err === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * err ** -0.2));
      h = Math.min(hMax, h * grow);
      return true;
    }
  }

  function extend(target: number): boolean {
    while (t < target) if (!step()) return false;
    return true;
  }

  /** The step that holds time s (binary search over the kept steps); -1 if none. */
  function stepAt(s: number): number {
    let lo = first, hi = tOld.length - 1;
    if (hi < lo || s < tOld[lo]) return -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (tOld[mid] <= s) lo = mid;
      else hi = mid - 1;
    }
    return s <= tOld[lo] + hs[lo] ? lo : -1;
  }

  function at(s: number): Vec {
    if (s === t0 && !forgotten) return y0.slice();
    if (s > t) extend(s);
    // a run that stopped still answers for every time its accepted steps cover (Codex review 3)
    if (s > t || s < tMin) return new Array<number>(n).fill(NaN);
    const j = stepAt(s);
    if (j < 0) return new Array<number>(n).fill(NaN);
    const th = (s - tOld[j]) / hs[j], th1 = 1 - th, c = cont[j];
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = c[i] + th * (c[n + i] + th1 * (c[2 * n + i] + th * (c[3 * n + i] + th1 * c[4 * n + i])));
    return out;
  }

  function crossings(g: (t: number, y: Vec) => number, ta: number, tb: number, o: { samples?: number; xtol?: number } = {}): Crossing[] {
    const out: Crossing[] = [];
    if (!(tb > ta)) return out;
    extend(tb);
    const inner = Math.max(0, Math.round(o.samples ?? 4)), xtol = o.xtol ?? 1e-13;
    const G = (s: number) => g(s, at(s));
    // the sign at a sample; an exact zero takes the sign just after it, so a crossing through an exact zero is counted
    // once, in the interval where g leaves 0, and a touch (from above or below) is no crossing (Codex review 1)
    const nudge = (s: number) => s + 1e-9 * Math.max(1, Math.abs(s));
    const sign = (s: number, v: number) => (v !== 0 ? (v < 0 ? -1 : 1) : G(nudge(s)) < 0 ? -1 : 1);
    let prevT = ta, prevG = G(ta), prevS = sign(ta, prevG);
    const visit = (s: number) => {
      if (s <= prevT) return;
      const gs = G(s);
      if (!Number.isFinite(gs) || !Number.isFinite(prevG)) { prevT = s; prevG = gs; prevS = Number.isFinite(gs) ? sign(s, gs) : 0; return; }
      const ss = sign(s, gs);
      if (ss !== prevS && prevS !== 0) {
        // the root: at s when g is exactly 0 there; else inside (prevT, s] by Brent — from just after prevT when g was 0
        // there (that zero's own crossing, if any, was counted where g left it)
        const tc = gs === 0 ? s : brent(G, prevG === 0 ? nudge(prevT) : prevT, s, { xtol }).x;
        if (tc > ta && tc <= tb && Number.isFinite(tc)) out.push({ t: tc, dir: ss > 0 ? 1 : -1 });
      }
      prevT = s;
      prevG = gs;
      prevS = ss;
    };
    for (let j = Math.max(first, stepAt(ta)); j < tOld.length && tOld[j] < tb; j++) {
      const a = tOld[j], h = hs[j];
      for (let k = 1; k <= inner; k++) visit(Math.min(tb, a + (k * h) / (inner + 1)));
      visit(Math.min(tb, a + h));
    }
    return out;
  }

  function integrate(fn: (t: number, y: Vec) => number, ta: number, tb: number, pieces = 1): number {
    if (!(tb > ta)) return tb === ta ? 0 : -integrate(fn, tb, ta, pieces);
    extend(tb);
    if (tb > t || ta < tMin) return NaN;
    const np = Math.max(1, Math.round(pieces));
    let sum = 0;
    const piece = (a: number, b: number) => {
      const c = (a + b) / 2, r = (b - a) / 2;
      for (let i = 0; i < 5; i++) {
        const s = c + r * GL_X[i];
        sum += GL_W[i] * r * fn(s, at(s));
      }
    };
    const j0 = stepAt(ta);
    if (j0 < 0) return NaN;
    for (let j = j0; j < tOld.length && tOld[j] < tb; j++) {
      const a = Math.max(ta, tOld[j]), b = Math.min(tb, tOld[j] + hs[j]);
      for (let k = 0; k < np; k++) piece(a + ((b - a) * k) / np, a + ((b - a) * (k + 1)) / np);
    }
    return sum;
  }

  function forget(s: number) {
    while (first < tOld.length - 1 && tOld[first] + hs[first] < s) {
      cont[first] = new Float64Array(0);
      first++;
      forgotten = true;
      tMin = tOld[first];
    }
    // compact now and then, so a long run does not keep a long array of empties
    if (first > 4096) {
      tOld.splice(0, first);
      hs.splice(0, first);
      cont.splice(0, first);
      first = 0;
    }
  }

  return {
    t0,
    get tEnd() { return t; },
    get stop() { return stop; },
    get steps() { return steps; },
    extend, at, crossings, integrate, forget,
  };
}
