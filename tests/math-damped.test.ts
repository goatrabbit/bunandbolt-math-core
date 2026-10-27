// The damped oscillator in closed form (src/lib/math/damped.ts) and the rigorous enclosures of π and ln
// (src/lib/math/precise.ts).
import { describe, expect, test } from 'vitest';
import { dampedMotion, dampingOf, dampingUnder, decayRate, roots } from '../src/lib/math/damped';
import { lnBounds, piBounds } from '../src/lib/math/precise';
import { Q } from '../src/lib/math/rational';
import { readTypedQ } from '../src/lib/ui/typed';

/** RK4 on x″ + 2ζx′ + x = 0 (the G05 check script's reference). */
function rk4(z: number, x0: number, v0: number, tEnd: number, dt = 1e-4) {
  let x = x0, v = v0;
  const f = (x: number, v: number) => [v, -2 * z * v - x];
  const n = Math.round(tEnd / dt);
  for (let i = 0; i < n; i++) {
    const [a1, b1] = f(x, v), [a2, b2] = f(x + (dt / 2) * a1, v + (dt / 2) * b1), [a3, b3] = f(x + (dt / 2) * a2, v + (dt / 2) * b2), [a4, b4] = f(x + dt * a3, v + dt * b3);
    x += (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
    v += (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
  }
  return [x, v];
}
const q = (s: string) => { const r = readTypedQ(s); if (!r.ok) throw new Error(s); return r.q; };

describe('damped oscillator', () => {
  test('the regime is decided on the rational, not its double', () => {
    expect(dampingOf(q('1')).regime).toBe('critical');
    expect(dampingOf(q('1.000')).regime).toBe('critical');
    expect(dampingOf(q('10/10')).regime).toBe('critical');
    const below = dampingOf(q('0.99999999999999999'));
    expect(below.regime).toBe('under');
    expect(below.zeta).toBe(1); // its double
    expect(below.root).toBeCloseTo(Math.sqrt(2e-17), 20); // 1 − ζ² formed exactly: ω_d ≈ 4.47e−9, not 0
    const above = dampingOf(q('1.00000000000000001'));
    expect(above.regime).toBe('over');
    expect(above.root).toBeGreaterThan(0);
    expect(() => dampingOf(Q.ZERO)).toThrow();
    // a ratio so near 1 that √(1 − ζ²) underflows: the exact regime, critical's motion, no NaN (Codex review 4)
    const huge = 10n ** 650n, m = dampedMotion(dampingOf(Q.of(huge - 1n, huge)));
    expect(m.damping.regime).toBe('under');
    expect(m.x(1)).toBeCloseTo(2 * Math.exp(-1), 15);
    expect(m.v(1)).toBeCloseTo(-Math.exp(-1), 15);
  });
  test('closed forms match RK4 in all three regimes, from rest and with a push', () => {
    for (const z of ['0.1', '0.6', '1', '3']) for (const [x0, v0] of [[1, 0], [0.3, -0.8]]) {
      const m = dampedMotion(dampingOf(q(z)), x0, v0);
      const [x, v] = rk4(Number(z), x0, v0, 7);
      expect(Math.abs(m.x(7) - x)).toBeLessThan(1e-9);
      expect(Math.abs(m.v(7) - v)).toBeLessThan(1e-9);
    }
  });
  test('across the overdamped switch (st = 1) the two forms agree', () => {
    for (const z of [1.0001, 1.5, 3, 20]) {
      const d = dampingOf(Q.fromDouble(z)), m = dampedMotion(d, 1, 0), t = 1 / d.root;
      const exp = (tt: number) => m.modes![0].a * Math.exp(m.modes![0].lambda * tt) + m.modes![1].a * Math.exp(m.modes![1].lambda * tt);
      const cosh = (tt: number) => Math.exp(-z * tt) * (Math.cosh(d.root * tt) + z * Math.sinh(d.root * tt) / d.root);
      expect(Math.abs(exp(t) - cosh(t)) / Math.abs(cosh(t))).toBeLessThan(1e-12);
      expect(Math.abs(m.x(t * (1 - 1e-12)) - m.x(t * (1 + 1e-12)))).toBeLessThan(1e-12);
    }
  });
  test('next to critical the three forms meet (no cancellation)', () => {
    const crit = dampedMotion(dampingOf(Q.ONE));
    for (const s of ['0.999999999999', '1.000000000001', '0.99999999999999999', '1.00000000000000001']) {
      const m = dampedMotion(dampingOf(q(s)));
      for (const t of [0.5, 3, 12, 40]) expect(Math.abs(m.x(t) - crit.x(t))).toBeLessThan(1e-9);
    }
  });
  test('large ζ and long times: no NaN, no overflow', () => {
    const m = dampedMotion(dampingOf(Q.of(100)));
    for (const t of [0, 1e-3, 1, 100, 900, 5000]) {
      expect(Number.isFinite(m.x(t))).toBe(true);
      expect(m.x(t)).toBeGreaterThanOrEqual(0);
    }
    expect(m.x(900)).toBeCloseTo(Math.exp(-900 / (100 + Math.sqrt(9999))) * (100 + Math.sqrt(9999)) / (2 * Math.sqrt(9999)), 12);
    const small = dampedMotion(dampingOf(Q.of(1, 100)));
    expect(Number.isFinite(small.x(2000))).toBe(true);
  });
  test('roots: product 1, the circle when underdamped, the slow root without cancellation', () => {
    const [[a, b]] = roots(dampingOf(Q.of(3, 5)));
    expect(a).toBeCloseTo(-0.6, 15);
    expect(b).toBeCloseTo(0.8, 15);
    expect(a * a + b * b).toBeCloseTo(1, 15);
    const [[s], [f]] = roots(dampingOf(Q.of(3)));
    expect(s).toBeCloseTo(-(3 - Math.sqrt(8)), 15);
    expect(f).toBeCloseTo(-(3 + Math.sqrt(8)), 14);
    expect(s * f).toBeCloseTo(1, 14);
    const [[s1000]] = roots(dampingOf(Q.of(1000)));
    expect(s1000).toBeCloseTo(-1 / (1000 + Math.sqrt(999999)), 18);
    expect(roots(dampingOf(Q.ONE))).toEqual([[-1, 0], [-1, 0]]);
  });
  test('decay rate: ζ up to critical, 1/(ζ + s) after, largest at 1', () => {
    expect(decayRate(dampingOf(Q.of(1, 5)))).toBe(0.2);
    expect(decayRate(dampingOf(Q.ONE))).toBe(1);
    expect(decayRate(dampingOf(Q.of(3)))).toBeCloseTo(0.1716, 4);
    let best = { z: 0, s: 0 };
    for (let i = 1; i <= 5000; i++) { const z = Q.of(i, 1000), s = decayRate(dampingOf(z)); if (s > best.s) best = { z: z.toNumber(), s }; }
    expect(best).toEqual({ z: 1, s: 1 });
  });
  test('the envelope R·e^(−ζt) bounds x and touches it; from rest R = 1/ω_d', () => {
    const d = dampingOf(Q.of(1, 10)), m = dampedMotion(d);
    let worst = -Infinity, touch = 0;
    for (let t = 0; t < 40; t += 1e-3) { worst = Math.max(worst, Math.abs(m.x(t)) - m.envelope(t)); touch = Math.max(touch, Math.abs(m.x(t)) / m.envelope(t)); }
    expect(worst).toBeLessThanOrEqual(1e-12);
    expect(touch).toBeCloseTo(1, 5);
    expect(m.envelope(0)).toBeCloseTo(1 / d.root, 15);
  });
  test('a symbol built from its own formulas behaves like the rational next to it', () => {
    const L = Math.log(100), zs = L / Math.hypot(Math.PI, L), wd = Math.PI / Math.hypot(Math.PI, L);
    const m = dampedMotion(dampingUnder(zs, wd));
    // the first extremum is at π/ω_d and is exactly −e^(−ζπ/ω_d) = −1/100
    expect(m.x(Math.PI / wd)).toBeCloseTo(-0.01, 15);
    expect(Math.abs(m.v(Math.PI / wd))).toBeLessThan(1e-16);
  });
});

describe('precise constants', () => {
  test('π: the enclosure holds the known digits and Math.PI, and is narrow', () => {
    const [lo, hi] = piBounds(200);
    // the first 62 decimals: the true value lies in [digits, digits + 10⁻⁶²], which must meet the enclosure
    const digits = Q.of(314159265358979323846264338327950288419716939937510582097494459n, 10n ** 62n);
    expect(lo.cmp(digits.add(Q.of(1n, 10n ** 62n))) <= 0 && digits.cmp(hi) <= 0).toBe(true);
    expect(Math.abs(lo.toNumber() - Math.PI)).toBeLessThan(1e-15);
    expect(hi.sub(lo).cmp(Q.of(1n, 1n << 200n))).toBeLessThan(0);
  });
  test('ln: ln 10 holds its known digits; ln 100 = 2 ln 10; agrees with Math.log', () => {
    const [lo, hi] = lnBounds(Q.of(10), 200);
    const digits = Q.of(230258509299404568401799145468436420760110148862877297603332790n, 10n ** 62n);
    expect(lo.cmp(digits.add(Q.of(1n, 10n ** 62n))) <= 0 && digits.cmp(hi) <= 0).toBe(true);
    const [l100, h100] = lnBounds(Q.of(100), 200);
    // the two enclosures of ln 100 = 2·ln 10 overlap
    expect(l100.cmp(hi.add(hi)) <= 0 && lo.add(lo).cmp(h100) <= 0).toBe(true);
    expect(Math.abs(l100.toNumber() - Math.log(100))).toBeLessThan(1e-15);
    expect(h100.sub(l100).toNumber()).toBeLessThan(1e-55);
    for (const x of [Q.of(1, 3), Q.of(7, 5), Q.of(12345, 7), Q.of(1, 1000000)]) {
      const [a, b] = lnBounds(x, 120);
      expect(a.cmp(b)).toBeLessThan(0);
      expect(Math.abs(a.toNumber() - Math.log(x.toNumber()))).toBeLessThan(1e-14);
    }
    expect(() => lnBounds(Q.ZERO)).toThrow();
  });
});
