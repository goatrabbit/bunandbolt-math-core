// The damped oscillator x″ + 2ζx′ + x = 0 in closed form (time in units of 1/ω₀), for the oscillation family: G05's
// three regimes, G06's transient, G07's points on the line Δ = ω₀². Exact in the sense of the model: every value is a
// closed-form expression evaluated in double precision, never a solver's output.
//
// The regime is decided exactly. A damping ratio typed or linked is a rational (`dampingOf(q)`): ζ = 1 compares
// integers, so 0.99999999999999999 is underdamped though its double is 1; 1 − ζ² and ζ² − 1 are formed as rationals and
// rounded once, so ω_d and s stay accurate next to ζ = 1. A ratio the page knows as a symbol (G05's ζ*) is built with
// `dampingUnder(ζ, ω_d)` from its own exact formulas.
//
// The formulas are written so that nothing cancels or overflows across the range a page allows (ζ from about 10⁻³ to
// 10³, any t ≥ 0): the slow overdamped root is −1/(ζ + s), never −ζ + s; the overdamped motion uses cosh / sinh while
// st ≤ 1 and the two exponentials after, where the faster one is at most e⁻² of the slower.
import { Q } from './rational';

export type Regime = 'under' | 'critical' | 'over';

export interface Damping {
  /** ζ as a double */
  zeta: number;
  regime: Regime;
  /** ω_d = √(1 − ζ²) when underdamped, s = √(ζ² − 1) when overdamped, 0 when critical */
  root: number;
}

/** The damping ratio from an exact rational: the regime by comparing with 1, the root from 1 − ζ² (or ζ² − 1) formed
 *  exactly. ζ must be positive. */
export function dampingOf(q: Q): Damping {
  if (q.sign() <= 0) throw new RangeError('damping ratio must be positive');
  const c = q.cmp(Q.ONE);
  if (c === 0) return { zeta: 1, regime: 'critical', root: 0 };
  const q2 = q.mul(q);
  const gap = c < 0 ? Q.ONE.sub(q2) : q2.sub(Q.ONE);
  return { zeta: q.toNumber(), regime: c < 0 ? 'under' : 'over', root: gap.sqrtToNumber() };
}

/** An underdamped ratio the page holds as a symbol, with ω_d from its own formula (both doubles, 0 < ζ < 1). */
export function dampingUnder(zeta: number, omegaD: number): Damping {
  return { zeta, regime: 'under', root: omegaD };
}

/** The characteristic roots λ = −ζ ± √(ζ² − 1) as [re, im] pairs: underdamped the conjugate pair (upper first);
 *  critical −1 twice; overdamped the slow root −1/(ζ + s) first, then the fast one −(ζ + s). */
export function roots(d: Damping): [[number, number], [number, number]] {
  if (d.regime === 'under') return [[-d.zeta, d.root], [-d.zeta, -d.root]];
  if (d.regime === 'critical') return [[-1, 0], [-1, 0]];
  const big = d.zeta + d.root;
  return [[-1 / big, 0], [-big, 0]];
}

/** The asymptotic decay rate σ = −max Re λ: ζ up to critical, 1/(ζ + √(ζ² − 1)) after; largest (= 1) at ζ = 1. */
export function decayRate(d: Damping): number {
  return d.regime === 'over' ? 1 / (d.zeta + d.root) : d.zeta;
}

export interface DampedMotion {
  readonly damping: Damping;
  /** position at time t ≥ 0 */
  x(t: number): number;
  /** velocity at time t ≥ 0 */
  v(t: number): number;
  /** underdamped: the envelope R·e^{−ζt} (|x| ≤ it, touching it once a half cycle); otherwise NaN */
  envelope(t: number): number;
  /** overdamped: the two modes x = a₁e^{λ₁t} + a₂e^{λ₂t}, slow first; otherwise null */
  readonly modes: { lambda: number; a: number }[] | null;
}

/** The motion from x(0) = x0, x′(0) = v0. */
export function dampedMotion(d: Damping, x0 = 1, v0 = 0): DampedMotion {
  const z = d.zeta, r = d.root;
  // sin(rt)/r and sinh(rt)/r → t when the root underflowed to 0 (a ratio within ~10⁻³⁰⁰ of 1: the regime stays the
  // exact one, the motion is critical's to double precision; Codex implementation review 4)
  const sinc = (t: number) => (r === 0 ? t : Math.sin(r * t) / r), sinhc = (t: number) => (r === 0 ? t : Math.sinh(r * t) / r);
  const c = v0 + z * x0; // the coefficient of sin / sinh (times the root), and of t when critical
  if (d.regime === 'under') {
    const B = c / r, R = Math.hypot(x0, B);
    return {
      damping: d,
      x: (t) => Math.exp(-z * t) * (x0 * Math.cos(r * t) + c * sinc(t)),
      v: (t) => Math.exp(-z * t) * (v0 * Math.cos(r * t) - (x0 + z * v0) * sinc(t)),
      envelope: (t) => R * Math.exp(-z * t),
      modes: null,
    };
  }
  if (d.regime === 'critical') {
    const k = v0 + x0;
    return {
      damping: d,
      x: (t) => Math.exp(-t) * (x0 + k * t),
      v: (t) => Math.exp(-t) * (v0 - k * t),
      envelope: () => NaN,
      modes: null,
    };
  }
  const big = z + r, l1 = -1 / big, l2 = -big;
  // x = a₁e^{λ₁t} + a₂e^{λ₂t} with a₁ = (x0(ζ + s) + v0)/(2s), a₂ = −(x0/(ζ + s) + v0)/(2s) (ζ − s written as 1/(ζ + s),
  // Codex design review 1). They are large and opposite when s is small, so this form waits until st > 1, where the
  // fast exponential is at most e⁻² of the slow one and their difference loses nothing that matters
  const a1 = (x0 * big + v0) / (2 * r), a2 = -(x0 / big + v0) / (2 * r);
  return {
    damping: d,
    x: (t) => (r * t <= 1
      ? Math.exp(-z * t) * (x0 * Math.cosh(r * t) + c * sinhc(t))
      : a1 * Math.exp(l1 * t) + a2 * Math.exp(l2 * t)),
    v: (t) => (r * t <= 1
      ? Math.exp(-z * t) * (v0 * Math.cosh(r * t) - (x0 + z * v0) * sinhc(t))
      : l1 * a1 * Math.exp(l1 * t) + l2 * a2 * Math.exp(l2 * t)),
    envelope: () => NaN,
    modes: [{ lambda: l1, a: a1 }, { lambda: l2, a: a2 }],
  };
}
