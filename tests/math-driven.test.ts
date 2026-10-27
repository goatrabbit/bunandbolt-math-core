// The driven damped oscillator in closed form (src/lib/math/driven.ts): the periodic solution and its exact
// decisions, the whole motion from any state against RK4, the beat form at and near r = 1, the transient's bound, the
// energy account and the zeros of v.
import { describe, expect, test } from 'vitest';
import {
  displacementPeak, drivenMotion, energyOver, halfPower, negativeShare, steadyPower, steadyState, steadyWork, velocityZeros,
} from '../src/lib/math/driven';
import { Q } from '../src/lib/math/rational';
import { readTypedQ } from '../src/lib/ui/typed';

const q = (s: string) => { const r = readTypedQ(s); if (!r.ok) throw new Error(s); return r.q; };

/** RK4 on x″ + 2ζx′ + x = cos(rt + θ₀). */
function rk4(z: number, r: number, th: number, x0: number, v0: number, tEnd: number, dt = 1e-3) {
  let x = x0, v = v0;
  const f = (t: number, x: number, v: number) => [v, Math.cos(r * t + th) - 2 * z * v - x];
  const n = Math.round(tEnd / dt);
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    const [a1, b1] = f(t, x, v), [a2, b2] = f(t + dt / 2, x + (dt / 2) * a1, v + (dt / 2) * b1);
    const [a3, b3] = f(t + dt / 2, x + (dt / 2) * a2, v + (dt / 2) * b2), [a4, b4] = f(t + dt, x + dt * a3, v + dt * b3);
    x += (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
    v += (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
  }
  return [x, v];
}

describe('the periodic solution', () => {
  test('amplitude and lag, from the exact rationals', () => {
    const s = steadyState(q('0.05'), Q.ONE)!;
    expect(s.A).toBeCloseTo(10, 12);
    expect(s.lag).toBe(Math.PI / 2); // exactly: 1 − r² is exactly 0
    for (const z of ['0.001', '0.3', '1', '2']) expect(steadyState(q(z), Q.ONE)!.lag).toBe(Math.PI / 2);
    const lo = steadyState(q('0.05'), q('0.5'))!, hi = steadyState(q('0.05'), Q.of(2))!;
    expect(lo.A).toBeCloseTo(1.33038, 5);
    expect((lo.lag * 180) / Math.PI).toBeCloseTo(3.814, 3);
    expect(hi.A).toBeCloseTo(0.332595, 6);
    expect((hi.lag * 180) / Math.PI).toBeCloseTo(176.186, 3);
    // a cos θ + b sin θ is A cos(θ − φ)
    expect(Math.hypot(lo.a, lo.b)).toBeCloseTo(lo.A, 14);
    expect(Math.atan2(lo.b, lo.a)).toBeCloseTo(lo.lag, 14);
  });
  test('undamped: a periodic solution off resonance (lag 0 or π), none at r = 1 exactly', () => {
    expect(steadyState(Q.ZERO, Q.ONE)).toBeNull();
    expect(steadyState(Q.ZERO, q('10/10'))).toBeNull();
    const near = steadyState(Q.ZERO, q('0.99999999999999999'))!; // its double is 1; 1 − r² is formed exactly
    expect(near.A / 5e16).toBeCloseTo(1, 14);
    expect(near.lag).toBe(0);
    expect(steadyState(Q.ZERO, Q.of(2))!.lag).toBe(Math.PI);
    expect(steadyState(Q.ZERO, Q.of(2))!.A).toBeCloseTo(1 / 3, 15);
  });
  test('the displacement peak √(1 − 2ζ²), decided exactly', () => {
    const p = displacementPeak(q('0.05'))!;
    expect(p.r).toBeCloseTo(0.9975, 4);
    expect(p.A).toBeCloseTo(10.0125, 4);
    expect(p.radicand.eq(q('0.995'))).toBe(true);
    const p2 = displacementPeak(q('0.2'))!;
    expect(p2.r).toBeCloseTo(0.9592, 4);
    expect(p2.A).toBeCloseTo(2.5516, 4);
    expect(displacementPeak(q('0.7071'))).not.toBeNull(); // 2·0.7071² = 0.99998… < 1
    expect(displacementPeak(q('0.7072'))).toBeNull();
    const edge = displacementPeak(q('0.70710678'))!; // 1 − 2ζ² = 3.36e−9 formed exactly: r_p = 5.79e−5
    expect(edge.r.toPrecision(3)).toBe('0.0000579');
    // against the radicand formed in integers (the double 1 − 2·0.70710678² keeps only six digits of it)
    expect(edge.r / Math.sqrt(Number(10n ** 16n - 2n * 70710678n ** 2n) / 1e16)).toBeCloseTo(1, 14);
    expect(Math.abs(edge.r - Math.sqrt(1 - 2 * 0.70710678 ** 2))).toBeGreaterThan(1e-13);
    expect(displacementPeak(Q.ZERO)).toEqual({ r: 1, A: Infinity, radicand: Q.ONE });
    // the scan agrees: no amplitude above the peak's
    for (const z of [0.05, 0.2, 0.5]) {
      const pk = displacementPeak(Q.fromDouble(z))!;
      let best = 0;
      for (let r = 0.01; r < 2; r += 1e-4) best = Math.max(best, 1 / Math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2));
      expect(best).toBeLessThanOrEqual(pk.A + 1e-12);
      expect(best).toBeCloseTo(pk.A, 6);
    }
  });
  test('half power: width exactly 2ζ, the power peak at r = 1', () => {
    for (const z of [0.01, 0.05, 0.2, 1, 2]) {
      const [r1, r2] = halfPower(z);
      expect(r2 - r1).toBeCloseTo(2 * z, 14);
      const P = (r: number) => steadyPower(z, r, steadyState(Q.fromDouble(z), Q.fromDouble(r))!.A);
      expect(P(r1)).toBeCloseTo(P(1) / 2, 12);
      expect(P(r2)).toBeCloseTo(P(1) / 2, 12);
      expect(P(1)).toBeCloseTo(1 / (4 * z), 12);
    }
    // a scan of the power curve finds 0.1 and 0.4 (the check script's way)
    const width = (z: number, f: (r: number) => number) => {
      let peak = 0;
      for (let r = 0.01; r < 3; r += 1e-5) peak = Math.max(peak, f(r));
      let lo = 0, hi = 0;
      for (let r = 0.01; r < 3; r += 1e-6) { if (!lo && f(r) >= peak / 2) lo = r; if (lo && !hi && f(r) < peak / 2) hi = r; }
      return hi - lo;
    };
    const M = (r: number, z: number) => 1 / Math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2);
    expect(width(0.05, (r) => (r * M(r, 0.05)) ** 2)).toBeCloseTo(0.1, 5);
    expect(width(0.2, (r) => (r * M(r, 0.2)) ** 2)).toBeCloseTo(0.4, 5);
    // on displacement² it would not be 2ζ (not used by the page)
    expect(width(0.05, (r) => M(r, 0.05) ** 2)).toBeCloseTo(0.100252, 5);
    expect(width(0.2, (r) => M(r, 0.2) ** 2)).toBeCloseTo(0.418698, 5);
  });
  test('the steady work per cycle πA sin φ = 2πζrA², and the share of negative power |½ − φ/π|', () => {
    for (const [z, r] of [[0.05, 1], [0.05, 0.5], [0.3, 2], [1, 0.7]]) {
      const s = steadyState(Q.fromDouble(z), Q.fromDouble(r))!;
      expect(steadyWork(z, r, s.A)).toBeCloseTo(Math.PI * s.A * Math.sin(s.lag), 12);
      // the share, against a sampled cycle
      let neg = 0;
      const N = 100000;
      for (let i = 0; i < N; i++) { const th = (2 * Math.PI * (i + 0.5)) / N; if (Math.cos(th) * -r * s.A * Math.sin(th - s.lag) < 0) neg++; }
      expect(neg / N).toBeCloseTo(negativeShare(s.lag), 4);
    }
    expect(negativeShare(steadyState(q('0.05'), q('0.5'))!.lag)).toBeCloseTo(0.4788, 4);
    expect(negativeShare(steadyState(q('0.05'), Q.of(2))!.lag)).toBeCloseTo(0.4788, 4);
    expect(steadyWork(0.05, 1, 10)).toBeCloseTo(10 * Math.PI, 12);
  });
});

describe('the whole motion', () => {
  test('agrees with RK4 from any state and phase, in all three regimes and undamped', () => {
    const cases: [string, string, number, number, number][] = [
      ['0.05', '1', 0, 0, 0], ['0.01', '0.9', 0, 0, 0], ['0', '1', 0.3, 0.2, -0.4], ['0', '0.9', 1, 0.5, 0.1],
      ['0.3', '2', 2, 1, 1], ['0', '1.0000001', 0.7, 0, 0], ['1', '0.8', 0.4, -0.3, 0.2], ['1.5', '1.3', 5, 0.6, -1],
    ];
    for (const [z, r, th, x0, v0] of cases) {
      const m = drivenMotion(q(z), q(r), th, x0, v0);
      expect(m.x(0)).toBeCloseTo(x0, 13);
      expect(m.v(0)).toBeCloseTo(v0, 13);
      const [x, v] = rk4(Number(z), Number(r), th, x0, v0, 40);
      expect(Math.abs(m.x(40) - x)).toBeLessThan(1e-8);
      expect(Math.abs(m.v(40) - v)).toBeLessThan(1e-8);
    }
  });
  test('undamped at r = 1 from rest: (t/2) sin t, a swing π longer each cycle; continuous in r', () => {
    const m = drivenMotion(Q.ZERO, Q.ONE, 0, 0, 0);
    for (const t of [0.5, 3, 17.2, 60]) expect(m.x(t)).toBeCloseTo((t / 2) * Math.sin(t), 12);
    // the largest |x| over ten periods: 30.63 (the check script)
    let mx = 0;
    for (let t = 0; t <= 20 * Math.PI; t += 1e-3) mx = Math.max(mx, Math.abs(m.x(t)));
    expect(mx).toBeCloseTo(30.63, 2);
    const near = drivenMotion(Q.ZERO, Q.ONE.add(Q.of(1n, 10n ** 12n)), 0.4, 0.1, 0.2), at1 = drivenMotion(Q.ZERO, Q.ONE, 0.4, 0.1, 0.2);
    // the true difference is about Δr·t²: no cancellation adds to it
    for (const t of [1, 10, 100]) expect(Math.abs(near.x(t) - at1.x(t))).toBeLessThan(1e-12 * (1 + t * t));
  });
  test('restarting from the state at t₁ continues the same motion (the page cuts a run into pieces)', () => {
    for (const z of ['0.05', '0', '1', '1.5']) {
      const r = q('0.9'), whole = drivenMotion(q(z), r, 0.3, 0.2, -0.1), t1 = 7.3;
      const rest = drivenMotion(q(z), r, 0.3 + 0.9 * t1, whole.x(t1), whole.v(t1));
      for (const s of [0, 1, 5, 20]) {
        expect(Math.abs(rest.x(s) - whole.x(t1 + s))).toBeLessThan(1e-11);
        expect(Math.abs(rest.v(s) - whole.v(t1 + s))).toBeLessThan(1e-11);
      }
    }
  });
  test('the transient bound holds from then on and never grows', () => {
    for (const [z, r, x0, v0] of [['0.05', '1', 0, 0], ['1', '0.6', 2, -3], ['1', '0.6', -0.5, 4], ['1.8', '2', 1, 1], ['0.3', '0.4', 3, 0]] as const) {
      const m = drivenMotion(q(z), q(r), 0, x0, v0), s = m.steady!;
      const xp = (t: number) => s.a * Math.cos(m.r * t) + s.b * Math.sin(m.r * t);
      let prev = Infinity;
      for (let t = 0; t < 60; t += 0.05) {
        const b = m.transientBound(t);
        expect(b).toBeLessThanOrEqual(prev + 1e-15);
        prev = b;
        for (const u of [t, t + 0.7, t + 3]) expect(Math.abs(m.x(u) - xp(u))).toBeLessThanOrEqual(b + 1e-12);
      }
    }
    expect(drivenMotion(Q.ZERO, q('0.9'), 0, 0, 0).transientBound(3)).toBeNaN();
  });
  test('next to ζ = 1 the bound stays meaningful (the envelope\'s coefficient is huge there: Codex design review 11)', () => {
    for (const z of ['0.99999999999999999', '1.00000000000000001', '0.999', '1.001']) {
      const m = drivenMotion(q(z), Q.ONE, 0, 0, 0), s = m.steady!;
      const xp = (t: number) => s.a * Math.cos(t) + s.b * Math.sin(t);
      expect(m.transientBound(10) / s.A).toBeLessThan(0.01);
      for (let t = 0; t < 30; t += 0.25) for (const u of [t, t + 0.4, t + 2]) expect(Math.abs(m.x(u) - xp(u))).toBeLessThanOrEqual(m.transientBound(t) + 1e-12);
    }
  });
  test('energy account: W_in = W_d + ΔE, the check script\'s cycles', () => {
    const m = drivenMotion(q('0.05'), Q.ONE, 0, 0, 0), T = 2 * Math.PI;
    let worst = 0;
    for (let k = 0; k < 80; k++) {
      const e = energyOver(m, k * T, (k + 1) * T);
      worst = Math.max(worst, Math.abs(e.win - e.wd - e.dE) / Math.max(1, Math.abs(e.win)));
    }
    expect(worst).toBeLessThan(1e-10);
    const third = energyOver(m, 2 * T, 3 * T);
    expect(third.win).toBeCloseTo(17.033, 3);
    expect(third.wd).toBeCloseTo(9.2932, 3);
    expect(third.dE).toBeCloseTo(7.7398, 3);
    const late = energyOver(m, 79 * T, 80 * T);
    expect(late.win).toBeCloseTo(10 * Math.PI, 2);
    expect(late.wd).toBeCloseTo(10 * Math.PI, 2);
    // heavy damping and a slow drive: the fast mode e^{−(ζ + s)t} sets the pieces (Codex design review 5)
    const h = drivenMotion(q('2'), q('0.2'), 0, 0, 0), eh = energyOver(h, 0, T / 0.2);
    expect(Math.abs(eh.win - eh.wd - eh.dE) / Math.max(1, Math.abs(eh.win))).toBeLessThan(1e-10);
    // undamped: no dissipation, everything stored
    const u = drivenMotion(Q.ZERO, Q.ONE, 0, 0, 0), eu = energyOver(u, 0, 3 * T);
    expect(eu.wd).toBe(0);
    expect(eu.win).toBeCloseTo(eu.dE, 10);
  });
  test('the zeros of v: where x has its extrema, found on the closed form', () => {
    const m = drivenMotion(Q.ZERO, Q.ONE, 0, 0, 0); // x = (t/2) sin t, v = (sin t + t cos t)/2
    const z = velocityZeros(m, 0, 20);
    for (const c of z) expect(Math.abs(Math.sin(c.t) + c.t * Math.cos(c.t))).toBeLessThan(1e-11);
    expect(z.length).toBe(6); // tan t = −t has one root in each (kπ − π/2, kπ + π/2), k = 1 … 6, below 20
    // cut into frames, the same zeros
    const parts: number[] = [];
    for (let a = 0; a < 20; a += 0.37) parts.push(...velocityZeros(m, a, Math.min(20, a + 0.37)).map((c) => c.t));
    expect(parts.length).toBe(z.length);
    parts.forEach((t, i) => expect(t).toBeCloseTo(z[i].t, 12));
    // an exact zero at an end is counted once (in the piece where v leaves it)
    const rest = drivenMotion(q('0.05'), Q.ONE, Math.PI, 1, 0); // v(0) = 0 exactly, then falling
    expect(velocityZeros(rest, 0, 0.5).length).toBe(0);
  });
  test('the record of peaks per drive cycle (the check script): 2.117 … 5.782 … 9.533; beats 9.07 / 3.06', () => {
    const peaks = (z: string, r: string, n: number) => {
      const m = drivenMotion(q(z), q(r), 0, 0, 0), T = (2 * Math.PI) / Number(r), out: number[] = [];
      for (let k = 0; k < n; k++) {
        let best = Math.max(Math.abs(m.x(k * T)), Math.abs(m.x((k + 1) * T)));
        for (const c of velocityZeros(m, k * T, (k + 1) * T)) best = Math.max(best, Math.abs(m.x(c.t)));
        out.push(best);
      }
      return out;
    };
    const p = peaks('0.05', '1', 12);
    expect(p[0]).toBeCloseTo(2.11736, 4);
    expect(p[2]).toBeCloseTo(5.78196, 4);
    expect(p.findIndex((v) => v >= 5)).toBe(2);
    expect(p.findIndex((v) => v >= 9.5)).toBe(9);
    expect(p[9]).toBeCloseTo(9.53335, 4);
    const b = peaks('0.01', '0.9', 30);
    expect(Math.max(...b.slice(0, 12))).toBeCloseTo(9.074, 2);
    expect(Math.min(...b.slice(6, 16))).toBeCloseTo(3.064, 2);
    expect(steadyState(q('0.01'), q('0.9'))!.A).toBeCloseTo(5.2397, 4);
    expect(1 / (Math.sqrt(1 - 1e-4) - 0.9)).toBeCloseTo(10.005, 3); // the beat period in T₀
  });
  test('from rest at r = 1: the power dips to −0.0524 at 0.285 T₀, then stays positive in the steady state', () => {
    const m = drivenMotion(q('0.05'), Q.ONE, 0, 0, 0);
    let mn = 0, at = 0;
    for (let t = 0; t < 2 * Math.PI; t += 1e-4) { const P = m.force(t) * m.v(t); if (P < mn) { mn = P; at = t; } }
    expect(mn).toBeCloseTo(-0.05238, 4);
    expect(at / (2 * Math.PI)).toBeCloseTo(0.2852, 3);
    const s = steadyState(q('0.05'), Q.ONE)!;
    for (let th = 0; th < 2 * Math.PI; th += 0.01) expect(Math.cos(th) * -s.A * Math.sin(th - s.lag)).toBeGreaterThanOrEqual(-1e-12);
  });
});
