import { describe, expect, it } from 'vitest';
import { adaptiveSimpson, diskIntegral, gaussLegendre, rectIntegral, gaussLegendreEst, gaussLegendreNodes } from '../src/lib/math/quad';
import { erf } from '../src/lib/math/special';

describe('adaptiveSimpson', () => {
  it('smooth integrands to the tolerance, with an honest error estimate', () => {
    const cases: [(x: number) => number, number, number, number][] = [
      [Math.sin, 0, Math.PI, 2],
      [Math.exp, 0, 1, Math.E - 1],
      [(x) => Math.exp(-x * x), -5, 5, Math.sqrt(Math.PI) * erf(5)],
      [(x) => 1 / (1 + x * x), 0, 1, Math.PI / 4],
      [(x) => x * Math.sin(1 / x), 0.01, 1, 0.37852917099769854], // fast oscillation at the left end (mpmath, 30 digits)
    ];
    for (const [f, a, b, want] of cases) {
      const r = adaptiveSimpson(f, a, b, { tol: 1e-11 });
      expect(r.ok).toBe(true);
      expect(Math.abs(r.value - want)).toBeLessThan(1e-9);
      expect(Math.abs(r.value - want)).toBeLessThanOrEqual(Math.max(r.error * 10, 1e-14));
    }
  });
  it('a square-root endpoint still converges', () => {
    const r = adaptiveSimpson(Math.sqrt, 0, 1, { tol: 1e-12 });
    expect(Math.abs(r.value - 2 / 3)).toBeLessThan(1e-9);
  });
  it('reports ok: false at a singularity instead of a confident number', () => {
    expect(adaptiveSimpson((x) => 1 / x, 0, 1).ok).toBe(false);
    expect(adaptiveSimpson((x) => 1 / Math.sqrt(Math.abs(x - 0.3)), 0, 1, { maxDepth: 12 }).ok).toBe(false);
  });
  it('empty and reversed intervals', () => {
    expect(adaptiveSimpson(Math.exp, 2, 2).value).toBe(0);
    expect(Math.abs(adaptiveSimpson(Math.exp, 1, 0).value + (Math.E - 1))).toBeLessThan(1e-10);
  });
});

describe('Gauss–Legendre', () => {
  it('nodes and weights: closed forms for n = 2, 3 and the weights sum to 2', () => {
    const g2 = gaussLegendreNodes(2);
    expect(Math.abs(g2.x[1] - 1 / Math.sqrt(3))).toBeLessThan(3e-16);
    expect(Math.abs(g2.w[0] - 1)).toBeLessThan(1e-15);
    const g3 = gaussLegendreNodes(3);
    expect(Math.abs(g3.x[2] - Math.sqrt(3 / 5))).toBeLessThan(1e-15);
    expect(Math.abs(g3.x[1])).toBeLessThan(1e-16);
    expect(Math.abs(g3.w[1] - 8 / 9)).toBeLessThan(1e-15);
    for (const n of [1, 5, 10, 20, 64]) {
      const { w } = gaussLegendreNodes(n);
      expect(Math.abs(w.reduce((s, v) => s + v, 0) - 2)).toBeLessThan(1e-14);
    }
  });
  it('exact for polynomials of degree 2n − 1', () => {
    // ∫₀² x⁹ dx = 2¹⁰/10 with n = 5
    expect(Math.abs(gaussLegendre((x) => x ** 9, 0, 2, 5) - 102.4)).toBeLessThan(1e-12);
    // degree 2n is not exact: ∫₋₁¹ x¹⁰ = 2/11
    expect(Math.abs(gaussLegendre((x) => x ** 10, -1, 1, 5) - 2 / 11)).toBeGreaterThan(1e-3);
  });
  it('with an estimate: e^x to full precision, panels for a long range', () => {
    const r = gaussLegendreEst(Math.exp, 0, 1, 10);
    expect(r.ok).toBe(true);
    expect(Math.abs(r.value - (Math.E - 1))).toBeLessThan(1e-15);
    const s = gaussLegendreEst(Math.sin, 0, 100 * Math.PI, 10, 50);
    expect(Math.abs(s.value)).toBeLessThan(1e-11);
    expect(s.ok).toBe(true);
  });
  it('the estimate flags what it cannot resolve', () => {
    const r = gaussLegendreEst((x) => Math.sin(200 * x), 0, 1, 5, 1);
    expect(r.ok).toBe(false);
  });
});

describe('Codex review 2026-10-07: regressions', () => {
  it('#20 a wave that is 1 at the first five samples is not taken for a constant', () => {
    const r = adaptiveSimpson((x) => Math.cos(8 * Math.PI * x), 0, 1);
    expect(Math.abs(r.value)).toBeLessThan(1e-9);
  });
});

describe('Codex re-review 2026-10-07: regressions of the first fixes', () => {
  it('#20 converging on the last allowed level is converged, not out of budget', () => {
    const r = adaptiveSimpson((x) => x, 0, 1, { maxDepth: 4 });
    expect(r).toMatchObject({ value: 0.5, ok: true });
    expect(adaptiveSimpson((x) => Math.sqrt(x), 0, 1, { maxDepth: 3 }).ok).toBe(false); // genuinely out of budget
  });
});

describe('diskIntegral (E25: ∬ div F dA against the flux)', () => {
  it('polynomials and cos over a disc, anywhere', () => {
    // area; ∬ 3x² + 3y² over the disc about (1, 0) of radius 1 = π(3 + 1.5)
    expect(diskIntegral(() => 1, 2, -1, 0.7).value).toBeCloseTo(Math.PI * 0.49, 13);
    expect(diskIntegral((x, y) => 3 * x * x + 3 * y * y, 1, 0, 1).value).toBeCloseTo(4.5 * Math.PI, 12);
    // ∬ cos x over the disc of radius r about the origin = πr² · 2J₁(r)/r
    const J1 = (x: number) => { let s = 0, t = x / 2; for (let k = 0; k < 40; k++) { s += t; t *= -(x * x / 4) / ((k + 1) * (k + 2)); } return s; };
    const r = diskIntegral((x) => Math.cos(x), 0, 0, 1.5);
    expect(r.value).toBeCloseTo(Math.PI * 1.5 * 1.5 * 2 * J1(1.5) / 1.5, 12);
    expect(r.error).toBeLessThan(1e-9);
  });
  it('the nodes miss the centre: an integrable singularity there is integrated, not hit', () => {
    const r = diskIntegral((x, y) => 1 / Math.hypot(x, y), 0, 0, 1); // = 2π
    expect(Number.isFinite(r.value)).toBe(true);
    expect(r.value).toBeCloseTo(2 * Math.PI, 10);
  });
  it('a pole inside that is not integrable does not settle', () => {
    expect(diskIntegral((x, y) => 1 / (x * x + y * y), 0.3, 0.1, 1).ok).toBe(false);
  });
  it('a value missing only at finer nodes is missing, and the refinement stops (Codex review 2026-10-08)', () => {
    const r = rectIntegral((x) => (x > 0.99 ? NaN : 1 / (1.0001 - x)), 0, 1, 0, 1);
    expect(r.ok).toBe(false);
    expect(r.evaluations).toBeLessThan(1e6);
    expect(diskIntegral((x) => (x > 0.999 ? NaN : 1 / (1.0002 - x)), 0, 0, 1).ok).toBe(false);
  });
  it('rectangles: ∬ 3x² + 3y² over [0, 1]² = 2, over [0, ½]² = 0.125', () => {
    expect(rectIntegral((x, y) => 3 * x * x + 3 * y * y, 0, 1, 0, 1)).toMatchObject({ value: expect.closeTo(2, 13), ok: true });
    expect(rectIntegral((x, y) => 3 * x * x + 3 * y * y, 0, 0.5, 0, 0.5).value).toBeCloseTo(0.125, 14);
    expect(rectIntegral((x, y) => Math.cos(x) * Math.cos(y), -1, 1, -1, 1).value).toBeCloseTo(4 * Math.sin(1) ** 2, 12);
  });
});
