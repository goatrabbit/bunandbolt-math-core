// lib/math/cpoly.ts: complex polynomials (WB03). Expected values worked by hand or from review/artifacts/newton-fractal.
import { describe, expect, test } from 'vitest';
import { cabs, cdiv, csub, derivative, evaluate, fromRoots, rootsOf, rouche, smaleGamma, smaleGammaAtRoot, smaleRadius, smaleRadiusAtRoot, taylorAt, taylorRatiosAtRoot, SMALE_ALPHA, type Z } from '../src/lib/math/cpoly';

const close = (a: Z, b: Z, tol = 1e-12) => expect(cabs(csub(a, b))).toBeLessThan(tol);

describe('cpoly', () => {
  test('fromRoots: (z − 1)(z + 1)(z − i) = z³ − iz² − z + i', () => {
    const c = fromRoots([[1, 0], [-1, 0], [0, 1]]);
    [[0, 1], [-1, 0], [0, -1], [1, 0]].forEach((w, k) => close(c[k], w as Z));
  });
  test('evaluate and derivative: p = z³ − 2z + 2', () => {
    const c: Z[] = [[2, 0], [-2, 0], [0, 0], [1, 0]];
    close(evaluate(c, [0, 0]), [2, 0]);
    close(evaluate(c, [1, 0]), [1, 0]);
    close(evaluate(c, [0, 1]), [2, -3]); // −i − 2i + 2
    const d = derivative(c);
    close(evaluate(d, [1, 0]), [1, 0]); // 3 − 2
  });
  test('taylorAt: z³ at 1 is 1 + 3e + 3e² + e³', () => {
    const b = taylorAt([[0, 0], [0, 0], [0, 0], [1, 0]], [1, 0]);
    [[1, 0], [3, 0], [3, 0], [1, 0]].forEach((w, k) => close(b[k], w as Z));
  });
  test('rootsOf: the cube roots of unity, and z³ − 2z + 2', () => {
    const r = rootsOf(fromRoots([[1, 0], [-0.5, Math.sqrt(3) / 2], [-0.5, -Math.sqrt(3) / 2]]));
    expect(r.ok).toBe(true);
    for (const z of r.roots) expect(Math.abs(cabs(z) - 1)).toBeLessThan(1e-13);
    const f = rootsOf([[2, 0], [-2, 0], [0, 0], [1, 0]]);
    expect(f.ok).toBe(true);
    const real = f.roots.find((z) => Math.abs(z[1]) < 1e-9)!;
    expect(real[0]).toBeCloseTo(-1.76929235423863, 12);
    for (const z of f.roots) expect(cabs(evaluate([[2, 0], [-2, 0], [0, 0], [1, 0]], z))).toBeLessThan(1e-13);
  });
  test('Smale: γ and the radius (3 − √7)/(2γ)', () => {
    // z³ − 1 at 1: b₁ = 3, b₂ = 3, b₃ = 1 → γ = max(1, (1/3)^(1/2)) = 1
    const c3 = fromRoots([[1, 0], [-0.5, Math.sqrt(3) / 2], [-0.5, -Math.sqrt(3) / 2]]);
    expect(smaleGamma(c3, [1, 0])).toBeCloseTo(1, 12);
    expect(smaleRadius(c3, [1, 0])).toBeCloseTo(0.177124344467705, 12);
    // z² − 1 at 1: b₁ = 2, b₂ = 1 → γ = 1/2, radius 2α
    expect(smaleRadius(fromRoots([[1, 0], [-1, 0]]), [1, 0])).toBeCloseTo(2 * SMALE_ALPHA, 12);
    // a double root: p′ = 0, no radius
    expect(smaleRadius(fromRoots([[1, 0], [1, 0], [-1, 0]]), [1, 0])).toBe(0);
  });
  test('at a root, from the root differences: the same ratios as the coefficients when the roots are apart', () => {
    const roots: Z[] = [[1, 0], [-0.7, 0.4], [0.2, -1.1], [-0.3, 0.9]];
    const c = fromRoots(roots);
    roots.forEach((r, i) => {
      const b = taylorAt(c, r), { a, err } = taylorRatiosAtRoot(roots, i);
      for (let k = 1; k < b.length; k++) close(a[k], cdiv(b[k], b[1]), 1e-12);
      expect(err[2]).toBeLessThan(1e-12);
      expect(smaleGammaAtRoot(roots, i)).toBeGreaterThanOrEqual(smaleGamma(c, r) * (1 - 1e-12));
      expect(smaleRadiusAtRoot(roots, i)).toBeCloseTo(smaleRadius(c, r), 10);
    });
    // z³ − 1 at 1: A₂ = 1, A₃ = 1/3, γ = 1 (rounded up)
    const z3: Z[] = [[1, 0], [-0.5, Math.sqrt(3) / 2], [-0.5, -Math.sqrt(3) / 2]];
    expect(smaleGammaAtRoot(z3, 0)).toBeGreaterThanOrEqual(1);
    expect(smaleGammaAtRoot(z3, 0)).toBeLessThan(1 + 1e-11);
  });
  test('Rouché: z² has two zeros within 0.1 of 0 and none counted singly; z − 0.05 has one', () => {
    expect(rouche([[0, 0], [0, 0], [1, 0]], 2, 0.1)).toBe(true);
    expect(rouche([[0, 0], [0, 0], [1, 0]], 1, 0.1)).toBe(false);
    expect(rouche([[-0.05, 0], [1, 0]], 1, 0.1)).toBe(true);
    expect(rouche([[-0.05, 0], [1, 0]], 1, 0.01)).toBe(false);
  });
});
