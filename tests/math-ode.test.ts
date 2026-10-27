import { describe, expect, it } from 'vitest';
import { apply2, dopri5, euler, expm2, flow2, FLOW2, rk4, rk4Step, type M2 } from '../src/lib/math/ode';

const last = <T>(a: T[]) => a[a.length - 1];

describe('rk4', () => {
  it('y′ = y: e to 1e-9 at h = 0.01, and fourth order (halving h divides the error by ~16)', () => {
    const err = (steps: number) => Math.abs(last(rk4((_t, y) => [y[0]], 0, [1], 1, steps).y)[0] - Math.E);
    expect(err(100)).toBeLessThan(1e-9);
    const ratio = err(20) / err(40);
    expect(ratio).toBeGreaterThan(15);
    expect(ratio).toBeLessThan(17);
  });
  it('returns every point, the last at t1 exactly', () => {
    const s = rk4((_t, y) => [-y[0]], 0, [1], 2, 8);
    expect(s.t.length).toBe(9);
    expect(last(s.t)).toBe(2);
  });
  it('D27: y′ = x − y through (0, 1) reaches y(2) = 1 + 2e⁻² = 1.270671', () => {
    const exact = 1 + 2 * Math.exp(-2);
    expect(exact.toFixed(6)).toBe('1.270671'); // D27's printed value
    const s = rk4((x, y) => [x - y[0]], 0, [1], 2, 200);
    expect(Math.abs(last(s.y)[0] - exact)).toBeLessThan(1e-10);
    // D27's Euler table has error −0.271 at h = 1; RK4 at the same h (two steps, by hand: 3/4 then 41/32) is off by 0.011
    const f = (x: number, y: number[]) => [x - y[0]];
    expect(rk4Step(f, 0, [1], 1)[0]).toBe(0.75);
    expect(rk4Step(f, 1, [0.75], 1)[0]).toBe(1.28125);
  });
});

describe('dopri5', () => {
  it('meets its tolerance on y′ = y and lands exactly on t1', () => {
    const s = dopri5((_t, y) => [y[0]], 0, [1], 1, { rtol: 1e-10, atol: 1e-12 });
    expect(s.stop).toBe('done');
    expect(last(s.t)).toBe(1);
    expect(Math.abs(last(s.y)[0] - Math.E) / Math.E).toBeLessThan(1e-9);
  });
  it('integrates backwards too', () => {
    const s = dopri5((_t, y) => [y[0]], 1, [Math.E], 0, { rtol: 1e-10, atol: 1e-12 });
    expect(s.stop).toBe('done');
    expect(last(s.t)).toBe(0);
    expect(Math.abs(last(s.y)[0] - 1)).toBeLessThan(1e-9);
  });
  it('harmonic oscillator: back to the start after 10 periods', () => {
    const s = dopri5((_t, [x, v]) => [v, -x], 0, [1, 0], 20 * Math.PI, { rtol: 1e-10, atol: 1e-12 });
    expect(s.stop).toBe('done');
    const [x, v] = last(s.y);
    expect(Math.hypot(x - 1, v)).toBeLessThan(1e-7);
  });
  it('D27: logistic y′ = y(1 − y) through (0, 0.1) is 1/(1 + 9e⁻ˣ), so y(ln 9) = 0.5', () => {
    const s = dopri5((_t, y) => [y[0] * (1 - y[0])], 0, [0.1], Math.log(9), { rtol: 1e-11, atol: 1e-13 });
    expect(Math.abs(last(s.y)[0] - 0.5)).toBeLessThan(1e-10);
  });
  it('hMax bounds the step so a drawn curve has enough points', () => {
    const s = dopri5((_t, y) => [y[0]], 0, [1], 1, { hMax: 0.05 });
    for (let i = 1; i < s.t.length; i++) expect(s.t[i] - s.t[i - 1]).toBeLessThanOrEqual(0.05 + 1e-15);
    expect(s.t.length).toBeGreaterThanOrEqual(21);
  });

  // the stop rules: each one is reported, never connected past
  it("blow-up: y′ = y² from y(0) = 1 stops before x = 1 with 'blew-up'", () => {
    const s = dopri5((_t, y) => [y[0] * y[0]], 0, [1], 2);
    expect(s.stop).toBe('blew-up');
    expect(last(s.t)).toBeLessThan(1);
    expect(last(s.y)[0]).toBeGreaterThan(1000); // |f| = y² passed 1e6
    // up to there it follows 1/(1 − x)
    for (let i = 0; i < s.t.length; i++) expect(Math.abs(s.y[i][0] * (1 - s.t[i]) - 1)).toBeLessThan(1e-5);
  });
  it("left the box: the first point outside is kept and the stop is 'left-box'", () => {
    const s = dopri5(() => [1], 0, [0], 10, { inside: (y) => Math.abs(y[0]) <= 5, hMax: 0.5 });
    expect(s.stop).toBe('left-box');
    expect(last(s.y)[0]).toBeGreaterThan(5);
    expect(s.y[s.y.length - 2][0]).toBeLessThanOrEqual(5);
  });
  it("a jump the step cannot resolve: 'step-underflow' at the jump", () => {
    const s = dopri5((t, y) => [t < 1 ? Math.cos(y[0]) : 1e4 + Math.cos(y[0])], 0, [0], 2, { rtol: 1e-12, atol: 1e-14, maxRate: 1e9 });
    expect(s.stop).toBe('step-underflow');
    expect(Math.abs(last(s.t) - 1)).toBeLessThan(1e-6);
  });
  it("NaN in the field: 'not-finite', at the start or where it appears", () => {
    expect(dopri5(() => [NaN], 0, [1], 1).stop).toBe('not-finite');
    const s = dopri5((t) => [t < 0.5 ? 1 : NaN], 0, [0], 1);
    expect(s.stop).toBe('not-finite');
    expect(last(s.t)).toBeLessThan(0.5);
  });
  it("a barrier the page names: 'barrier', before the singular line (Codex #9)", () => {
    // y′ = 1/(t − 1)² from y = 1e12: the pole's effect is below the tolerance, so plain error control steps over it
    const s = dopri5((t) => [1 / (t - 1) ** 2], 0, [1e12], 2, { barrier: (t) => t - 1 });
    expect(s.stop).toBe('barrier');
    expect(last(s.t)).toBeLessThan(1);
    const u = dopri5((_t, y) => [-1 / y[0]], 0, [1], 2, { barrier: (_t, y) => y[0], maxRate: 1e12 });
    expect(['barrier', 'blew-up', 'step-underflow']).toContain(u.stop); // y = √(1 − 2t) reaches 0 at t = ½
    expect(last(u.t)).toBeLessThan(0.5 + 1e-6); // where the numerical solution ends: ½ to within the tolerance
  });
  it("the step budget: 'max-steps'", () => {
    const s = dopri5((_t, y) => [y[0]], 0, [1], 1, { hMax: 0.01, maxSteps: 10 });
    expect(s.stop).toBe('max-steps');
    expect(s.t.length).toBe(11);
  });
});

describe('expm2: exp(tA) for 2×2', () => {
  const close = (A: M2, B: M2, tol: number) => A.forEach((v, i) => expect(Math.abs(v - B[i])).toBeLessThanOrEqual(tol));
  const viaRk4 = (A: M2, t: number): M2 => {
    const f = (_t: number, y: number[]) => [A[0] * y[0] + A[1] * y[1], A[2] * y[0] + A[3] * y[1]];
    const c1 = last(rk4(f, 0, [1, 0], t, 2000).y), c2 = last(rk4(f, 0, [0, 1], t, 2000).y);
    return [c1[0], c2[0], c1[1], c2[1]];
  };
  it('rotation, shear, diagonal: closed forms', () => {
    close(expm2([0, -1, 1, 0], Math.PI / 2), [0, -1, 1, 0], 1e-15);
    close(expm2([0, 1, 0, 0], 3), [1, 3, 0, 1], 1e-15);
    close(expm2([2, 0, 0, -1], 0.5), [Math.E, 0, 0, Math.exp(-0.5)], 1e-15);
    close(expm2([1, 2, 3, 4], 0), [1, 0, 0, 1], 0);
  });
  it('agrees with RK4 on the three kinds (real, complex, repeated eigenvalues)', () => {
    for (const A of [[1, 2, 3, -4], [-0.5, -2, 1.5, -0.25], [1, 1, 0, 1], [0.3, 1e-9, 1e-9, 0.3]] as M2[]) {
      close(expm2(A, 1.3), viaRk4(A, 1.3), 1e-11);
    }
  });
  it('nearly equal eigenvalues lose nothing (δ = 1e-20)', () => {
    // A = [[1, 1e-10], [1e-10, 1]]: exp(tA) = eᵗ [[cosh(1e-10 t), sinh(1e-10 t)], …]
    const E = expm2([1, 1e-10, 1e-10, 1], 2);
    expect(Math.abs(E[0] / Math.exp(2) - 1)).toBeLessThan(1e-15);
    expect(Math.abs(E[1] / (Math.exp(2) * 2e-10) - 1)).toBeLessThan(1e-12);
  });
  it('real eigenvalues far apart: the decaying mode survives (Codex #10)', () => {
    const E = expm2([20, 0, 0, -20], 1);
    expect(Math.abs(E[0] / Math.exp(20) - 1)).toBeLessThan(1e-15);
    expect(Math.abs(E[3] / Math.exp(-20) - 1)).toBeLessThan(1e-15);
    expect([E[1], E[2]]).toEqual([0, 0]);
    // upper triangular [[−1, 1000], [0, −30]]: exact entries e^{−t}, e^{−30t} and 1000(e^{−t} − e^{−30t})/29
    const t = 2, F = expm2([-1, 1000, 0, -30], t);
    const want = [Math.exp(-t), (1000 * (Math.exp(-t) - Math.exp(-30 * t))) / 29, 0, Math.exp(-30 * t)];
    for (const k of [0, 1, 3]) expect(Math.abs(F[k] / want[k] - 1)).toBeLessThan(1e-14);
    expect(F[2]).toBe(0);
    // backwards in time it is the inverse
    const G = expm2([0.5, 2, 1, -0.7], -1.5), H = expm2([0.5, 2, 1, -0.7], 1.5);
    const I = [G[0] * H[0] + G[1] * H[2], G[0] * H[1] + G[1] * H[3], G[2] * H[0] + G[3] * H[2], G[2] * H[1] + G[3] * H[3]];
    I.forEach((v, k) => expect(Math.abs(v - (k === 0 || k === 3 ? 1 : 0))).toBeLessThan(1e-14));
  });
  it('exp(sA)·exp(tA) = exp((s + t)A) and the flow applies to vectors', () => {
    const A: M2 = [0.2, -1.1, 0.7, -0.3];
    const S = expm2(A, 0.4), T = expm2(A, 0.9), ST = expm2(A, 1.3);
    const prod: M2 = [S[0] * T[0] + S[1] * T[2], S[0] * T[1] + S[1] * T[3], S[2] * T[0] + S[3] * T[2], S[2] * T[1] + S[3] * T[3]];
    close(prod, ST, 1e-14);
    const v = apply2(ST, [1, 2]);
    const w = apply2(S, apply2(T, [1, 2]));
    expect(Math.hypot(v[0] - w[0], v[1] - w[1])).toBeLessThan(1e-14);
  });
});

describe('Codex re-review 2026-10-07: regressions of the first fixes', () => {
  it('#10 exp(tA) near the overflow threshold stays finite', () => {
    expect(expm2([709.5, 0, 0, 0], 1)[0]).toBe(Math.exp(709.5)); // 1.3549863193146326e308
    expect(expm2([0, 0, 0, 709.5], 1)[3]).toBe(Math.exp(709.5));
    const E = expm2([709, 0.5, 0, 708], 1); // upper triangular: off-diagonal 0.5(e^709 − e^708)
    expect(Number.isFinite(E[1])).toBe(true);
    expect(Math.abs(E[1] / (0.5 * (Math.exp(709) - Math.exp(708))) - 1)).toBeLessThan(1e-14);
  });
});

describe('flow2: dopri5 for many 2D points (the vector-field family)', () => {
  const out = new Float64Array(3);
  const rot = (x: number, y: number, o: Float64Array) => { o[0] = -y; o[1] = x; };
  it('agrees with dopri5 and the closed form, both directions, and lands on dt exactly', () => {
    out[2] = 0;
    expect(flow2(rot, 1, 0, Math.PI / 2, out, { rtol: 1e-10, atol: 1e-12 })).toBe(FLOW2.done);
    expect(out[0]).toBeCloseTo(0, 9);
    expect(out[1]).toBeCloseTo(1, 9);
    const pend = (x: number, y: number, o: Float64Array) => { o[0] = y; o[1] = -Math.sin(x); };
    out[2] = 0;
    flow2(pend, 1, 0.5, 3, out, { rtol: 1e-10, atol: 1e-12 });
    const ref = last(dopri5((_t, v) => [v[1], -Math.sin(v[0])], 0, [1, 0.5], 3, { rtol: 1e-11, atol: 1e-13 }).y);
    expect(Math.abs(out[0] - ref[0])).toBeLessThan(1e-8);
    expect(Math.abs(out[1] - ref[1])).toBeLessThan(1e-8);
    // and back again
    const [x1, y1] = [out[0], out[1]];
    out[2] = 0;
    flow2(pend, x1, y1, -3, out, { rtol: 1e-10, atol: 1e-12 });
    expect(Math.abs(out[0] - 1)).toBeLessThan(1e-8);
    expect(Math.abs(out[1] - 0.5)).toBeLessThan(1e-8);
  });
  it('one frame at a time gives the same as one call, and remembers its step', () => {
    const src = (x: number, y: number, o: Float64Array) => { o[0] = x; o[1] = y; };
    let x = 1, y = 0.5;
    out[2] = 0;
    for (let i = 0; i < 60; i++) { flow2(src, x, y, Math.LN2 / 60, out, { rtol: 1e-9, atol: 1e-12 }); x = out[0]; y = out[1]; }
    expect(x).toBeCloseTo(2, 8);
    expect(y).toBeCloseTo(1, 8);
    expect(out[2]).toBeGreaterThan(0);
  });
  it('E22: x′ = −1000x over one frame (1/120) decays like e^(−8.33), where fixed-step RK4 grows 132-fold', () => {
    const stiff = (x: number, y: number, o: Float64Array) => { o[0] = -1000 * x; o[1] = 0; };
    out[2] = 0;
    expect(flow2(stiff, 1, 0, 1 / 120, out, { rtol: 1e-7, atol: 1e-9 })).toBe(FLOW2.done);
    expect(Math.abs(out[0] - Math.exp(-1000 / 120))).toBeLessThan(1e-7);
    const z = -1000 / 120;
    expect(1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24).toBeCloseTo(131.877, 3);
  });
  it('stops and says why: a blow-up, the box, an undefined spot', () => {
    const sq = (x: number, y: number, o: Float64Array) => { o[0] = 0; o[1] = y * y; };
    out[2] = 0;
    expect(flow2(sq, 0, 1, 2, out)).toBe(FLOW2.blewUp); // y′ = y² from 1 blows up at t = 1
    expect(out[1]).toBeGreaterThan(1e2);
    out[2] = 0;
    expect(flow2(rot, 1, 0, 3, out, { inside: (x) => x > 0 })).toBe(FLOW2.leftBox);
    expect(out[0]).toBeLessThanOrEqual(0);
    const hole = (x: number, y: number, o: Float64Array) => { o[0] = 1; o[1] = Math.sqrt(1 - x); };
    out[2] = 0;
    expect([FLOW2.notFinite, FLOW2.stepUnderflow]).toContain(flow2(hole, 0, 0, 3, out));
    expect(out[0]).toBeLessThanOrEqual(1);
  });
});

describe('flow2 after the Codex implementation review (2026-10-07)', () => {
  it('a short call does not shrink the step carried to the next one', () => {
    const out = new Float64Array(3);
    out[2] = 0.1;
    flow2((x, y, o) => { o[0] = -y; o[1] = x; }, 1, 0, 0.001, out);
    expect(out[2]).toBeGreaterThanOrEqual(0.1);
  });
  it('stops at a barrier the page names, where error control alone steps over the pole', () => {
    const F = (x: number, _y: number, o: Float64Array) => { o[0] = 1; o[1] = 1e-12 / (x - 1) ** 2; };
    const out = new Float64Array(5);
    expect(flow2(F, 0.62, 1, 1.38, out)).toBe(FLOW2.done); // unseen without it
    out[2] = 0;
    expect(flow2(F, 0.62, 1, 1.38, out, { barrier: (x) => x - 1 })).toBe(FLOW2.barrier);
    expect(out[0]).toBeLessThan(1);
    expect(out[4]).toBeLessThan(0.38);
  });
});

describe('euler: the explicit rule as D27 runs it', () => {
  const xmy = (x: number, y: number) => x - y;
  it("y′ = x − y from (0, 1) to 2: 1 + 2(1 − h)^(2/h) at every halving (the direction draft's numbers)", () => {
    const want = [1, 1.125, 1.200226, 1.236134, 1.253578, 1.262168, 1.26643, 1.268553];
    for (let k = 0; k <= 7; k++) {
      const h = 1 / 2 ** k, r = euler(xmy, 0, 1, 2, h);
      expect(r.stop).toBe('done');
      expect(r.t.length).toBe(2 / h + 1);
      const v = r.y[r.y.length - 1];
      expect(v).toBeCloseTo(1 + 2 * (1 - h) ** (2 / h), 13);
      expect(v).toBeCloseTo(want[k], 6);
    }
  });
  it('the last step is shortened so it lands on the end; going left takes steps of −h', () => {
    const r = euler(xmy, 0, 1, 1, 0.3); // 0.3, 0.6, 0.9, then 0.1
    expect([...r.t]).toEqual([0, 0.3, 2 * 0.3, 3 * 0.3, 1]);
    const l = euler(xmy, 2, 0, 0, 0.5);
    expect([...l.t]).toEqual([2, 1.5, 1, 0.5, 0]);
    let y = 0;
    for (const t of [2, 1.5, 1, 0.5]) y += -0.5 * (t - y);
    expect(l.y[4]).toBe(y);
  });
  it('the start alone when the end is the start; one step when h is longer than the way', () => {
    expect([...euler(xmy, 1, 2, 1, 0.5).y]).toEqual([2]);
    const r = euler(xmy, 0, 1, 2, 4);
    expect([...r.t]).toEqual([0, 2]);
    expect(r.y[1]).toBe(1 + 2 * (0 - 1));
  });
  it('2/(1/128) is 256 steps, not 257, and a run past the budget is refused', () => {
    expect(euler(xmy, 0, 1, 2, 1 / 128).t.length).toBe(257);
    expect(euler(xmy, 0, 1, 0.3, 0.1).t.length).toBe(4); // 0.3/0.1 = 2.9999999999999996
    expect(euler(xmy, 0, 1, 100, 1e-5).stop).toBe('max-steps');
  });
  it("stops at the first value that is not finite: y′ = y² overflows, f without a value", () => {
    const r = euler((_x, y) => y * y, 0, 1, 2, 1 / 128);
    expect(r.stop).toBe('not-finite');
    expect(r.y.every(Number.isFinite)).toBe(true);
    expect(r.y[r.y.length - 1]).toBeGreaterThan(1e100);
    const u = euler((x, y) => -x / y, 1, 0, 2, 0.5);
    expect(u.stop).toBe('not-finite');
    expect(u.t.length).toBe(1);
  });
  it('y′ = 3y^(2/3) from (0, 0) stays at 0 for every h (the rule reads slope 0 there)', () => {
    for (const h of [1, 0.1, 0.001]) expect(euler((_x, y) => 3 * Math.cbrt(y) ** 2, 0, 0, 2, h).y.every((v) => v === 0)).toBe(true);
  });
  it('y′ = cos x from (0, 0) to π: the value is h itself', () => {
    for (const n of [1, 2, 4, 8, 64]) { const r = euler((x) => Math.cos(x), 0, 0, Math.PI, Math.PI / n); expect(r.y[r.y.length - 1]).toBeCloseTo(Math.PI / n, 12); }
  });
});
