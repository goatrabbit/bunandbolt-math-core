// The driven damped oscillator x″ + 2ζx′ + x = cos(rt + θ₀) (time in units of 1/ω₀, displacement in units of the
// static deflection x_st = F₀/k, r = ω/ω₀), for G06 resonance; a page that adds a force to G07's line Δ = ω₀² can use
// it as it is. Exact in the sense of the model: every value is a closed form evaluated in double precision.
//
// - The parameters are exact rationals (`Q`), so the questions a double cannot settle are decided exactly: ζ = 0 and
//   r = 1 (no periodic solution: the swing grows by π a cycle), whether the displacement has a peak (2ζ² < 1 compares
//   integers; equality never happens for a rational ζ since √2 is irrational), the free motion's regime (damped.ts).
//   1 − r², 2ζr and their sum of squares are formed as rationals and rounded once, so nothing cancels next to r = 1.
// - The whole motion from any state is the periodic solution plus G05's free motion (`dampedMotion`, all three
//   regimes). Undamped, the forced part from rest is written as a beat, [2 sin((r + 1)t/2 + θ₀)·S(t) − sin θ₀ sin t]
//   /(r + 1) with S(t) = sin((r − 1)t/2)/(r − 1): it never divides by 1 − r², and at r = 1 it is (t/2)sin(t + θ₀) −
//   ½ sin θ₀ sin t, the swing that grows without bound.
// - Damped, the periodic and the free parts are each of size about A ≤ 1/(2ζ) near r = 1 and cancel early on: the
//   relative error of x grows by that factor (500 at ζ = 0.001), which a page bounds through its smallest ζ.
import { Q } from './rational';
import { dampedMotion, dampingOf, dampingUnder, decayRate, type Damping, type DampedMotion } from './damped';
import { brent } from './roots';
import { gaussLegendre } from './quad';

const TWO = Q.of(2);

export interface Steady {
  /** the amplitude A = 1/√D, D = (1 − r²)² + (2ζr)² (in units of x_st) */
  A: number;
  /** the lag φ = atan2(2ζr, 1 − r²) in [0, π]: the displacement is A cos(θ − φ) when the force is cos θ */
  lag: number;
  /** the same solution as a cos θ + b sin θ: a = (1 − r²)/D, b = 2ζr/D (each rounded once from the exact rational) */
  a: number;
  b: number;
}

/** The periodic solution for exact ζ ≥ 0 and r > 0; null when ζ = 0 and r = 1, where none exists. At r = 1 the lag is
 *  exactly π/2 for every ζ > 0 (1 − r² is exactly 0). */
export function steadyState(zeta: Q, r: Q): Steady | null {
  const c = Q.ONE.sub(r.mul(r)), s = TWO.mul(zeta).mul(r);
  const D = c.mul(c).add(s.mul(s));
  if (D.isZero()) return null;
  return { A: 1 / D.sqrtToNumber(), lag: Math.atan2(s.toNumber(), c.toNumber()), a: c.div(D).toNumber(), b: s.div(D).toNumber() };
}

/** Where the displacement amplitude peaks: r_p = √(1 − 2ζ²), A_p = 1/(2ζ√(1 − ζ²)), and the exact radicand 1 − 2ζ².
 *  null when 2ζ² > 1 (A falls from r = 0 on; 2ζ² = 1 cannot happen for a rational ζ). Undamped: r = 1, A = ∞. */
export function displacementPeak(zeta: Q): { r: number; A: number; radicand: Q } | null {
  if (zeta.isZero()) return { r: 1, A: Infinity, radicand: Q.ONE };
  const g = Q.ONE.sub(TWO.mul(zeta).mul(zeta));
  if (g.sign() <= 0) return null;
  return { r: g.sqrtToNumber(), A: 1 / (2 * zeta.toNumber() * Q.ONE.sub(zeta.mul(zeta)).sqrtToNumber()), radicand: g };
}

/** The half-power points of the mean input power ζr²A² (peak 1/(4ζ) at r = 1 exactly): r = √(1 + ζ²) ∓ ζ, so the width
 *  is exactly 2ζ. The lower one is written 1/(√(1 + ζ²) + ζ), with no subtraction. */
export function halfPower(zeta: number): [number, number] {
  const h = Math.hypot(1, zeta);
  return [1 / (h + zeta), h + zeta];
}

/** The mean input power in the periodic state, ζr²A² (units of F₀·x_st·ω₀), and the work of the force over one drive
 *  cycle, 2πζrA² = πA·sin φ: equal to the dissipation, since nothing is stored. */
export const steadyPower = (zeta: number, r: number, A: number) => zeta * r * r * A * A;
export const steadyWork = (zeta: number, r: number, A: number) => 2 * Math.PI * zeta * r * A * A;

/** The share of a cycle in the periodic state where the force works against the motion (P = F·v < 0): |½ − φ/π|. */
export const negativeShare = (lag: number) => Math.abs(0.5 - lag / Math.PI);

export interface DrivenMotion {
  readonly zeta: number;
  readonly r: number;
  /** the periodic solution (null: ζ = 0 and r = 1) */
  readonly steady: Steady | null;
  /** the free motion's damping (ζ = 0: undamped, root 1) */
  readonly damping: Damping;
  /** position, velocity and the force cos(rt + θ₀) at time t ≥ 0 */
  x(t: number): number;
  v(t: number): number;
  force(t: number): number;
  /** A bound on |x − x_p| at every time from t on, non-increasing in t. Every regime has (|x_h0| + |c|·s)e^{−σs} with
   *  s = max(t, t*), c = v_h0 + ζx_h0, σ the decay rate and t* = 1/σ − |x_h0|/|c| where it peaks (|sin ω_d s|/ω_d ≤ s,
   *  sinh(qs)/q ≤ s·cosh(qs) ≤ s·e^{qs}); underdamped the smaller of it and R e^{−ζt} (the free part's envelope),
   *  overdamped of it and |a₁|e^{λ₁t} + |a₂|e^{λ₂t}. Next to ζ = 1 the envelope and the modes have huge, nearly
   *  cancelling coefficients and the first bound is the one that means something (Codex design review 11). NaN
   *  undamped (the free part never decays). */
  transientBound(t: number): number;
}

/** The motion from x(0) = x0, x′(0) = v0 under the force cos(rt + θ₀), exact ζ ≥ 0 and r > 0. */
export function drivenMotion(zeta: Q, r: Q, theta0: number, x0: number, v0: number): DrivenMotion {
  const rr = r.toNumber(), steady = steadyState(zeta, r);
  const force = (t: number) => Math.cos(rr * t + theta0);
  if (zeta.isZero()) {
    // the free part as given, plus the forced part from rest in its beat form (no 1/(1 − r²))
    const rm = r.sub(Q.ONE).toNumber(), rp = rr + 1;
    const S = (t: number) => (rm === 0 ? t / 2 : Math.sin((rm * t) / 2) / rm);
    const dS = (t: number) => Math.cos((rm * t) / 2) / 2;
    const s0 = Math.sin(theta0);
    return {
      zeta: 0, r: rr, steady, damping: dampingUnder(0, 1), force,
      x: (t) => x0 * Math.cos(t) + v0 * Math.sin(t) + (2 * Math.sin((rp * t) / 2 + theta0) * S(t) - s0 * Math.sin(t)) / rp,
      v: (t) => -x0 * Math.sin(t) + v0 * Math.cos(t) + (rp * Math.cos((rp * t) / 2 + theta0) * S(t) + 2 * Math.sin((rp * t) / 2 + theta0) * dS(t) - s0 * Math.cos(t)) / rp,
      transientBound: () => NaN,
    };
  }
  const { a, b } = steady!; // ζ > 0: D ≥ (2ζr)² > 0
  const xp = (t: number) => a * Math.cos(rr * t + theta0) + b * Math.sin(rr * t + theta0);
  const vp = (t: number) => rr * (b * Math.cos(rr * t + theta0) - a * Math.sin(rr * t + theta0));
  const d = dampingOf(zeta);
  const xh0 = x0 - xp(0), vh0 = v0 - vp(0);
  const free: DampedMotion = dampedMotion(d, xh0, vh0);
  const sigma = decayRate(d), c0 = Math.abs(xh0), k = Math.abs(vh0 + d.zeta * xh0);
  const tStar = k > 0 ? Math.max(0, 1 / sigma - c0 / k) : 0;
  const linear = (t: number) => { const s = Math.max(t, tStar); return (c0 + k * s) * Math.exp(-sigma * s); };
  let bound: (t: number) => number;
  if (d.regime === 'under') bound = (t) => Math.min(free.envelope(t), linear(t));
  else if (d.regime === 'critical') bound = linear;
  else {
    const [m1, m2] = free.modes!;
    bound = (t) => Math.min(Math.abs(m1.a) * Math.exp(m1.lambda * t) + Math.abs(m2.a) * Math.exp(m2.lambda * t), linear(t));
  }
  return {
    zeta: d.zeta, r: rr, steady, damping: d, force,
    x: (t) => xp(t) + free.x(t),
    v: (t) => vp(t) + free.v(t),
    transientBound: bound,
  };
}

/** The energy account over [ta, tb]: the work of the force ∫F·v, the dissipation ∫2ζv² (Gauss–Legendre, 8 points on
 *  pieces a quarter of the fastest period long, and short enough that v² of the fast overdamped mode, e^{−2(ζ + s)t},
 *  falls by at most e^{−1.5} across one: Codex design review 5) and the change of the stored energy ½(x² + v²) between
 *  the ends. The three are computed apart: W_in − W_d − ΔE is a check (it vanishes for the exact motion), not forced
 *  to zero. */
export function energyOver(m: DrivenMotion, ta: number, tb: number): { win: number; wd: number; dE: number } {
  if (!(tb > ta)) return { win: 0, wd: 0, dE: 0 };
  const d = m.damping, fast = d.regime === 'over' ? d.zeta + d.root : 1;
  const piece = Math.min(Math.PI / (2 * Math.max(m.r, 1)), 0.75 / fast);
  const panels = Math.max(2, Math.ceil((tb - ta) / piece));
  const win = gaussLegendre((t) => m.force(t) * m.v(t), ta, tb, 8, panels);
  const wd = m.zeta === 0 ? 0 : gaussLegendre((t) => 2 * m.zeta * m.v(t) ** 2, ta, tb, 8, panels);
  const E = (t: number) => 0.5 * (m.x(t) ** 2 + m.v(t) ** 2);
  return { win, wd, dE: E(tb) - E(ta) };
}

/** The grid on which `velocityZeros` looks for sign changes: a 24th of the faster half period (of the force and of
 *  the free motion), in the motion's own time. */
export const zeroGrid = (r: number) => Math.PI / Math.max(r, 1) / 24;

/** The times in (ta, tb] where v changes sign (the extrema of x), increasing. v is sampled at ta, at the multiples of
 *  `h` (default `zeroGrid`) inside and at tb, and each sign change is located by Brent's method on the closed form. An
 *  exact zero takes the sign just after it (as trajectory.ts), so a crossing through 0 is counted once and a touch not
 *  at all. Two zeros closer than one grid step can be missed together; there |v| ≤ |a|·|t − t*| around the extremum
 *  t*, so |x| anywhere within h of it differs from |x(t*)| by at most ½|a|h². */
export function velocityZeros(m: DrivenMotion, ta: number, tb: number, h = zeroGrid(m.r)): { t: number; dir: 1 | -1 }[] {
  const out: { t: number; dir: 1 | -1 }[] = [];
  if (!(tb > ta)) return out;
  const nudge = (s: number) => s + 1e-9 * Math.max(1, Math.abs(s));
  const sign = (s: number, v: number) => (v !== 0 ? (v < 0 ? -1 : 1) : m.v(nudge(s)) < 0 ? -1 : 1);
  let pt = ta, pv = m.v(ta), ps = sign(ta, pv);
  const visit = (s: number) => {
    const v = m.v(s), ss = sign(s, v);
    if (ss !== ps) {
      const tc = v === 0 ? s : brent(m.v, pv === 0 ? nudge(pt) : pt, s, { xtol: 1e-13 }).x;
      if (tc > ta && tc <= tb && Number.isFinite(tc)) out.push({ t: tc, dir: ss > 0 ? 1 : -1 });
    }
    pt = s;
    pv = v;
    ps = ss;
  };
  for (let k = Math.floor(ta / h) + 1; k * h < tb; k++) visit(k * h);
  visit(tb);
  return out;
}
