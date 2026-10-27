// The dense-output trajectory (src/lib/math/trajectory.ts) and the AGM / elliptic K (src/lib/math/elliptic.ts).
import { describe, expect, test } from 'vitest';
import { createTrajectory } from '../src/lib/math/trajectory';
import { agm, ellipK, ellipKc } from '../src/lib/math/elliptic';

const harmonic = (_t: number, y: number[]) => [y[1], -y[0]]; // y″ = −y

describe('elliptic', () => {
  test('AGM: known values and limits', () => {
    expect(agm(1, 1)).toBe(1);
    expect(agm(1, 0)).toBe(0);
    expect(agm(24, 6)).toBeCloseTo(13.458171481725615, 13); // Gauss's example
    expect(agm(1, Math.SQRT2)).toBeCloseTo(1.1981402347355923, 14); // Gauss's constant times √2 … 1/G = AGM(1, √2)
    expect(Number.isNaN(agm(-1, 1))).toBe(true);
  });
  test('K: K(0) = π/2, K(1/√2) = Γ(1/4)²/(4√π), the series near 0, ∞ at 1', () => {
    expect(ellipK(0)).toBeCloseTo(Math.PI / 2, 15);
    expect(ellipK(Math.SQRT1_2)).toBeCloseTo(1.854074677301372, 14);
    const k = 0.01;
    expect(ellipK(k)).toBeCloseTo((Math.PI / 2) * (1 + k * k / 4 + (9 / 64) * k ** 4 + (25 / 256) * k ** 6), 14);
    expect(ellipKc(0)).toBe(Infinity);
    // close to 1, the complementary modulus keeps the digits: K ≈ ln(4/k′) for small k′
    expect(ellipKc(1e-8)).toBeCloseTo(Math.log(4 / 1e-8), 6);
  });
});

describe('trajectory: dense output', () => {
  test('y″ = −y from (1, 0): at(t) matches (cos t, −sin t) between and at steps', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { rtol: 1e-10, atol: 1e-12, hMax: 0.2 });
    let worst = 0;
    for (let i = 0; i <= 2000; i++) {
      const t = i * 0.0137;
      const [x, v] = tr.at(t);
      worst = Math.max(worst, Math.abs(x - Math.cos(t)), Math.abs(v + Math.sin(t)));
    }
    expect(worst).toBeLessThan(1e-8);
    expect(tr.stop).toBeNull();
    expect(tr.at(0)).toEqual([1, 0]);
  });
  test('the same time asked in any order gives the same value (the steps do not depend on the questions)', () => {
    const a = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.1 });
    const b = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.1 });
    const t = 7.31;
    for (let i = 1; i < 100; i++) a.at(i * 0.0731); // a walks there in small asks, b in one
    expect(a.at(t)).toEqual(b.at(t));
  });
  test('a blow-up stops it and says why; beyond the stop at() is NaN', () => {
    const tr = createTrajectory((_t, y) => [y[0] * y[0]], 0, [1], { maxRate: 1e8 }); // y = 1/(1 − t)
    expect(tr.extend(2)).toBe(false);
    expect(tr.stop).toBe('blew-up');
    expect(tr.tEnd).toBeLessThan(1);
    expect(Number.isNaN(tr.at(1.5)[0])).toBe(true);
    expect(tr.at(0.5)[0]).toBeCloseTo(2, 7);
  });
  test('forget drops old steps: at() there is NaN, later times still work', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.05 });
    tr.at(20);
    tr.forget(10);
    expect(Number.isNaN(tr.at(1)[0])).toBe(true);
    expect(tr.at(15)[0]).toBeCloseTo(Math.cos(15), 8);
  });
});

describe('trajectory: crossings', () => {
  test('x = cos t crosses 0 at π/2 + nπ, alternating down and up', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.25 });
    const c = tr.crossings((_t, y) => y[0], 0, 10);
    expect(c.map((e) => e.dir)).toEqual([-1, 1, -1]);
    c.forEach((e, n) => expect(Math.abs(e.t - (Math.PI / 2 + n * Math.PI))).toBeLessThan(1e-9));
  });
  test('asked in pieces, the crossings are the same', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.25 });
    const whole = tr.crossings((_t, y) => y[0], 0, 20).map((e) => e.t);
    const parts: number[] = [];
    for (let a = 0; a < 20; a += 0.37) parts.push(...tr.crossings((_t, y) => y[0], a, Math.min(20, a + 0.37)).map((e) => e.t));
    expect(parts.length).toBe(whole.length);
    parts.forEach((t, i) => expect(t).toBeCloseTo(whole[i], 12));
  });
  test('the pendulum (rtol 10⁻¹²): over a minute, half periods from the crossings equal T₀/(2·AGM(1, cos(θ₀/2))) to 10⁻⁹ s', () => {
    const g = 9.81, L = 1, w0 = Math.sqrt(g / L);
    for (const deg of [10, 90, 170]) {
      const th0 = (deg * Math.PI) / 180;
      const tr = createTrajectory((_t, y) => [y[1], -(g / L) * Math.sin(y[0])], 0, [th0, 0], { rtol: 1e-12, atol: 1e-14, hMax: (2 * Math.PI) / w0 / 40 });
      const c = tr.crossings((_t, y) => y[0], 0, 60);
      const T = (2 * Math.PI) / w0 / agm(1, Math.cos(th0 / 2));
      for (let i = 1; i < c.length; i++) expect(Math.abs(c[i].t - c[i - 1].t - T / 2)).toBeLessThan(1e-9);
      c.forEach((e, i) => expect(Math.abs(e.t - T / 4 - (i * T) / 2)).toBeLessThan(1e-8)); // a minute later still within 10⁻⁸ s
    }
  });
});

describe('trajectory: the edge cases of Codex\'s implementation review', () => {
  const still = () => createTrajectory(() => [0], 0, [0], { h0: 1, hMax: 1 }); // y ≡ 0: only g's own time matters
  test('a crossing through an exact zero is counted once, whether asked whole or in pieces', () => {
    const g = (t: number) => 0.5 - t;
    const whole = still().crossings((t) => g(t), 0, 1, { samples: 1 });
    expect(whole.map((c) => [c.t, c.dir])).toEqual([[0.5, -1]]);
    const tr = still();
    const parts = [...tr.crossings((t) => g(t), 0, 0.5, { samples: 1 }), ...tr.crossings((t) => g(t), 0.5, 1, { samples: 1 })];
    expect(parts.map((c) => [c.t, c.dir])).toEqual([[0.5, -1]]);
    const up = still();
    expect([...up.crossings((t) => t - 0.5, 0, 0.5, { samples: 1 }), ...up.crossings((t) => t - 0.5, 0.5, 1, { samples: 1 })].map((c) => [c.t, c.dir])).toEqual([[0.5, 1]]);
  });
  test('a touch of zero from below or above is no crossing', () => {
    expect(still().crossings((t) => -((t - 0.5) ** 2), 0, 1, { samples: 1 })).toEqual([]);
    expect(still().crossings((t) => (t - 0.5) ** 2, 0, 1, { samples: 1 })).toEqual([]);
  });
  test('a run that stops after the step covering t still answers at t, on the first ask', () => {
    const tr = createTrajectory((t) => [2 * t], 0, [0], { h0: 1, hMax: 1, maxRate: 1 });
    expect(tr.at(0.5)[0]).toBeCloseTo(0.25, 12);
    expect(tr.stop).toBe('blew-up');
    expect(Number.isNaN(tr.at(1.5)[0])).toBe(true);
    const tr2 = createTrajectory((t) => [2 * t], 0, [0], { h0: 1, hMax: 1, maxRate: 1 });
    expect(tr2.integrate(() => 1, 0, 0.5)).toBeCloseTo(0.5, 12);
  });
  test('after forget and the compaction, the start is not answered again', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { h0: 0.001, hMax: 0.001 });
    tr.at(5);
    tr.forget(4.5);
    expect(Number.isNaN(tr.at(0)[0])).toBe(true);
    expect(Number.isNaN(tr.at(1)[0])).toBe(true);
    expect(tr.at(4.8)[0]).toBeCloseTo(Math.cos(4.8), 8);
  });
  test('AGM keeps its scale: huge and tiny arguments', () => {
    expect(agm(1e200, 1e200)).toBe(1e200);
    expect(agm(1e-200, 1e-200)).toBe(1e-200);
    expect(agm(24e100, 6e100) / 1e100).toBeCloseTo(13.458171481725615, 12);
  });
});

describe('trajectory: integrals on the dense output', () => {
  test('∫ x² over a period of x = cos t is π; over a part, the closed form', () => {
    const tr = createTrajectory(harmonic, 0, [1, 0], { hMax: 0.2 });
    expect(tr.integrate((_t, y) => y[0] * y[0], 0, 2 * Math.PI)).toBeCloseTo(Math.PI, 9);
    const F = (t: number) => t / 2 + Math.sin(2 * t) / 4;
    expect(tr.integrate((_t, y) => y[0] * y[0], 0.3, 4.1)).toBeCloseTo(F(4.1) - F(0.3), 9);
    expect(tr.integrate((_t, y) => y[0], 1, 1)).toBe(0);
  });
});

