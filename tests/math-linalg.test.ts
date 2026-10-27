import { describe, expect, it } from 'vitest';
import {
  cond2, cubicRoots, det2Q, det2Sign, det3Exact, det3Q, eig2, eig2Q, eig3, eig3Q, fitLine, inv2, inv3, leastSquares, project, rankQ,
  inv2Q, rref, solve2, solve2Exact, solve2Q, solve3, solve3Q, type M2, type M3, type Q2, type Q3, type V2, type V3,
} from '../src/lib/math/linalg';
import { parseExact, Q } from '../src/lib/math/rational';

// Cases cite review/artifacts/page-directions/F03.mjs (systems), F13.mjs (eigenvectors), F19.mjs (least squares).

const near = (a: number, b: number, tol = 1e-12) => expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol);
const nearV = (a: number[], b: number[], tol = 1e-12) => a.forEach((v, i) => near(v, b[i], tol));
/** same line through 0 (directions compared as unit vectors, up to sign) */
const sameDir = (u: V2 | V3, v: number[]) => {
  const unit = (a: number[]) => a.map((x) => x / Math.hypot(...a));
  const uu = unit(u), w = unit(v);
  const s = Math.sign(uu.reduce((acc, x, i) => acc + x * w[i], 0)) || 1;
  nearV(uu.map((x) => x * s), w);
};

describe('2×2 systems by case (F03)', () => {
  it('a unique crossing, exact for the numbers given', () => {
    const r = solve2([2, -1, 1, 1], [1, 5]); // 2·(2,1) + 3·(−1,1) = (1,5)
    expect(r.kind).toBe('unique');
    if (r.kind === 'unique') expect(r.x).toEqual([2, 3]);
  });
  it('nearly parallel lines: the true solution and a large condition number, not "no solution"', () => {
    const a = solve2([1, 1, 1, 1.01], [2, 2.01]);
    const b = solve2([1, 1, 1, 1.01], [2, 2.02]);
    expect(a.kind).toBe('unique');
    expect(b.kind).toBe('unique');
    if (a.kind === 'unique' && b.kind === 'unique') {
      nearV(a.x, [1, 1], 1e-12); // F03: (1, 1), and (0, 2) after moving the right side by 0.01
      nearV(b.x, [0, 2], 1e-12);
      expect(a.condition).toBeGreaterThan(300);
    }
  });
  it('singular: none, a line, or every point — decided exactly', () => {
    expect(solve2([1, 1, 2, 2], [2, 5]).kind).toBe('none'); // R2 − 2R1 = (0 = 1)
    const line = solve2([1, 1, 2, 2], [2, 4]); // R2 − 2R1 = (0 = 0)
    expect(line.kind).toBe('line');
    if (line.kind === 'line') {
      near(line.point[0] + line.point[1], 2);
      sameDir(line.direction, [1, -1]);
    }
    expect(solve2([0, 0, 0, 0], [0, 0]).kind).toBe('plane');
    expect(solve2([0, 0, 0, 0], [0, 1]).kind).toBe('none');
    // x + y = 2, x + (1 + e)y = 2 has (2, 0) for every e ≠ 0, however small (F03)
    for (const e of [1e-3, 1e-9, 2 ** -52]) {
      const r = solve2([1, 1, 1, 1 + e], [2, 2]);
      expect(r.kind).toBe('unique');
      if (r.kind === 'unique') expect(r.x).toEqual([2, 0]);
    }
  });
  it('exact singularity tests where floating point would round', () => {
    // [[1, 1/3], [3, 1]]: (1/3)·3 rounds to 1, but with the double nearest 1/3 the determinant is 5.55e-17, not 0
    expect(1 - (1 / 3) * 3).toBe(0);
    expect(det2Sign([1, 1 / 3, 3, 1])).toBe(1);
    expect(solve2([1, 1 / 3, 3, 1], [1, 3]).kind).toBe('unique');
    // whereas 0.3 = 0.6/2 and 0.2 = 2·0.1 exactly in binary: [[0.1, 0.3], [0.2, 0.6]] is exactly singular
    expect(det2Sign([0.1, 0.3, 0.2, 0.6])).toBe(0);
    expect(det2Sign([1, 3, 2, 6])).toBe(0);
    expect(inv2([1, 3, 2, 6])).toBeNull();
    const inv = inv2([4, 7, 2, 6])!;
    nearV(inv, [0.6, -0.7, -0.2, 0.4], 1e-16);
    near(cond2([1, 0, 0, 1e-8]), 1e8, 1e-6);
  });
});

describe('2×2 eigenvalues and eigenvectors by case (F13)', () => {
  it('distinct real: the default [[3, 1], [2, 2]] has 4 on (1, 1) and 1 on (1, −2)', () => {
    const e = eig2([3, 1, 2, 2]);
    expect(e.kind).toBe('distinct');
    if (e.kind === 'distinct') {
      nearV(e.values, [4, 1], 1e-15);
      sameDir(e.vectors[0], [1, 1]);
      sameDir(e.vectors[1], [1, -2]);
    }
  });
  it('symmetric, swap, Fibonacci, c = 0.04', () => {
    const cases: [M2, [number, number], V2, V2][] = [
      [[2, 1, 1, 2], [3, 1], [1, 1], [1, -1]],
      [[0, 1, 1, 0], [1, -1], [1, 1], [1, -1]],
      [[1, 1, 1, 0], [(1 + Math.sqrt(5)) / 2, (1 - Math.sqrt(5)) / 2], [(1 + Math.sqrt(5)) / 2, 1], [(1 - Math.sqrt(5)) / 2, 1]],
      [[1, 1, 0.04, 1], [1.2, 0.8], [1, 0.2], [1, -0.2]],
    ];
    for (const [A, values, v1, v2] of cases) {
      const e = eig2(A);
      expect(e.kind).toBe('distinct');
      if (e.kind === 'distinct') {
        nearV(e.values, values, 1e-15);
        sameDir(e.vectors[0], v1);
        sameDir(e.vectors[1], v2);
      }
    }
  });
  it('complex (rotation by 30°, c < 0), defective (shear), scalar', () => {
    const r = eig2([Math.cos(Math.PI / 6), -0.5, 0.5, Math.cos(Math.PI / 6)]);
    expect(r.kind).toBe('complex');
    if (r.kind === 'complex') {
      near(r.re, Math.cos(Math.PI / 6), 1e-16);
      near(r.im, 0.5, 1e-15);
    }
    expect(eig2([1, 1, -0.04, 1]).kind).toBe('complex');
    const s = eig2([1, 1, 0, 1]);
    expect(s.kind).toBe('defective');
    if (s.kind === 'defective') {
      expect(s.value).toBe(1);
      sameDir(s.vector, [1, 0]);
    }
    expect(eig2([2, 0, 0, 2])).toEqual({ kind: 'scalar', value: 2 });
  });
  it('the case is exact: 3·(−1/3) rounds to −1, but the matrix given has two distinct eigenvalues', () => {
    // A = [[2, 3], [c, 0]] with c the double nearest −1/3: ((a − d)/2)² + bc = 1 + 3c = 5.55e-17 > 0 exactly
    const A: M2 = [2, 3, -1 / 3, 0];
    expect(1 + 3 * (-1 / 3)).toBe(0); // what a floating discriminant would say
    const e = eig2(A);
    expect(e.kind).toBe('distinct');
    if (e.kind === 'distinct') {
      const r = Math.sqrt(Q.of(1).add(Q.of(3).mul(Q.fromDouble(-1 / 3))).toNumber());
      near(e.values[0] - e.values[1], 2 * r, 1e-22);
      // both eigenvectors satisfy A v = λ v to rounding
      e.vectors.forEach((v, k) => {
        const Av = [A[0] * v[0] + A[1] * v[1], A[2] * v[0] + A[3] * v[1]];
        nearV(Av, [e.values[k] * v[0], e.values[k] * v[1]], 1e-15);
      });
      expect(Math.abs(e.vectors[0][0] * e.vectors[1][1] - e.vectors[0][1] * e.vectors[1][0])).toBeGreaterThan(1e-9);
    }
  });
  it('nearly equal eigenvalues keep separate, correct eigenvectors ([[1, 1e-12], [1e-12, 1]])', () => {
    const e = eig2([1, 1e-12, 1e-12, 1]);
    expect(e.kind).toBe('distinct');
    if (e.kind === 'distinct') {
      sameDir(e.vectors[0], [1, 1]);
      sameDir(e.vectors[1], [1, -1]);
    }
  });
  it('a small eigenvalue next to a large one keeps its digits (from det / λ₁)', () => {
    // det = 1e8 × (the double nearest 1e-8) − 1 = 2.09e-17 exactly; floating point gives 0 or ±2.2e-16
    const e = eig2([1e8, 1, 1, 1e-8]);
    expect(e.kind).toBe('distinct');
    const det = Q.fromDouble(1e8).mul(Q.fromDouble(1e-8)).sub(Q.of(1)).toNumber();
    expect(det).toBeGreaterThan(2e-17);
    if (e.kind === 'distinct') expect(Math.abs((e.values[0] * e.values[1]) / det - 1)).toBeLessThan(1e-15);
  });
});

describe('3×3 (F03 three planes)', () => {
  it('row 3 = row 1 + row 2: det 0 exactly; b = (1, 0, 3) has no solution, b = (1, 0, 1) a line along (3, −1, −2)', () => {
    const A: M3 = [1, 1, 1, 1, -1, 2, 2, 0, 3];
    expect(det3Exact(A).isZero()).toBe(true);
    expect(inv3(A)).toBeNull();
    expect(solve3(A, [1, 0, 3])).toEqual({ kind: 'none', rank: 2 });
    const line = solve3(A, [1, 0, 1]);
    expect(line.kind).toBe('line');
    if (line.kind === 'line') {
      sameDir(line.direction.map((x) => x) as V3, [3, -1, -2]);
      const p = line.point;
      nearV([p[0] + p[1] + p[2], p[0] - p[1] + 2 * p[2], 2 * p[0] + 3 * p[2]], [1, 0, 1]);
    }
  });
  it('unique, plane and all-of-space cases', () => {
    const u = solve3([2, 0, 0, 0, 3, 0, 0, 0, 4], [2, 3, 4]);
    expect(u.kind).toBe('unique');
    if (u.kind === 'unique') expect(u.x).toEqual([1, 1, 1]);
    const p = solve3([1, 2, 3, 2, 4, 6, 3, 6, 9], [1, 2, 3]);
    expect(p.kind).toBe('plane');
    if (p.kind === 'plane') sameDir(p.normal, [1, 2, 3]);
    expect(solve3([0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0]).kind).toBe('space');
    const inv = inv3([2, 0, 0, 0, 4, 0, 0, 0, 8])!;
    expect(inv).toEqual([0.5, 0, 0, 0, 0.25, 0, 0, 0, 0.125]);
  });
  it('exact row reduction over Q', () => {
    const { R, pivots } = rref([[1, 2, 3], [2, 4, 7]]);
    expect(pivots).toEqual([0, 2]);
    expect(R.map((row) => row.map((q) => q.toNumber()))).toEqual([[1, 2, 0], [0, 0, 1]]);
  });
  it('cubic roots and 3×3 eigenvalues', () => {
    expect(cubicRoots(-6, 11, -6).real.map((x) => Math.round(x * 1e12) / 1e12)).toEqual([1, 2, 3]);
    const c = cubicRoots(0, 0, -1); // t³ = 1
    near(c.real[0], 1);
    near(c.complex!.re, -0.5);
    near(c.complex!.im, Math.sqrt(3) / 2);
    const e = eig3([2, 0, 0, 0, 3, 4, 0, 4, 9]);
    nearV(e.values, [1, 2, 11], 1e-12);
    sameDir(e.vectors[2]!, [0, 1, 2]);
  });
});

describe('projection and least squares (F19)', () => {
  it('best line through (0, 0), (1, 1), (2, 1): y = 1/6 + x/2, residuals (−1/6, 1/3, −1/6), SSE 1/6', () => {
    const f = fitLine([0, 1, 2], [0, 1, 1]);
    expect(f.unique).toBe(true);
    near(f.c, 1 / 6, 1e-15);
    near(f.m, 0.5, 1e-15);
    nearV(f.residuals, [-1 / 6, 1 / 3, -1 / 6], 1e-15);
    near(f.sse, 1 / 6, 1e-15);
    // the same through QR on the columns (1, 1, 1) and x
    const ls = leastSquares([[1, 1, 1], [0, 1, 2]], [0, 1, 1]);
    nearV(ls.coords, [1 / 6, 0.5], 1e-15);
    nearV(ls.e, [-1 / 6, 1 / 3, -1 / 6], 1e-15);
  });
  it("Strang's example: (0, 6), (1, 0), (2, 0) → 5 − 3t, residuals (1, −2, 1), SSE 6", () => {
    const f = fitLine([0, 1, 2], [6, 0, 0]);
    near(f.c, 5);
    near(f.m, -3);
    nearV(f.residuals, [1, -2, 1]);
    near(f.sse, 6);
  });
  it('centred formulas survive a large x offset', () => {
    const f = fitLine([1e8, 1e8 + 1, 1e8 + 2], [0, 1, 1]);
    near(f.m, 0.5, 1e-9);
    nearV(f.residuals, [-1 / 6, 1 / 3, -1 / 6], 1e-8);
  });
  it('all x equal: no unique line (decided exactly, even for 0.1 three times)', () => {
    for (const x of [1, 0.1]) {
      const f = fitLine([x, x, x], [1, 2, 4]);
      expect(f.unique).toBe(false);
      expect(f.c).toBeNaN();
      nearV(f.fitted, [7 / 3, 7 / 3, 7 / 3], 1e-15);
    }
  });
  it('projection onto (1, 1, 1) is the mean; dependent basis vectors still give the right p', () => {
    const p = project([0, 1, 1], [[1, 1, 1]]);
    nearV(p.p, [2 / 3, 2 / 3, 2 / 3], 1e-15);
    near(p.e.reduce((s, x) => s + x, 0), 0, 1e-15);
    const d = project([0, 1, 1], [[1, 1, 1], [2, 2, 2], [0, 1, 2]]);
    expect(d.independent).toBe(false);
    nearV(d.p, [1 / 6, 2 / 3, 7 / 6], 1e-15); // the fitted values of the best line
  });
  it('the error is orthogonal to every basis vector', () => {
    const basis = [[1, 2, 0, 1], [0, 1, 1, -1]];
    const r = project([3, -1, 2, 5], basis);
    for (const v of basis) near(v.reduce((s, x, i) => s + x * r.e[i], 0), 0, 1e-14);
  });
});

describe('Codex review 2026-10-07: regressions', () => {
  it('#11 exactly independent columns that floating QR merges: the exact projection', () => {
    const r = project([1, 0], [[3, 4], [0.6, 0.8]]); // det of these doubles is 2⁻⁵²: they span the plane
    expect(r.independent).toBe(true);
    expect(r.p).toEqual([1, 0]);
    expect(r.e).toEqual([0, 0]);
  });
  it('#12 eig2 with entries whose squares under- or overflow', () => {
    const e = eig2([1e-200, 0, 0, 2e-200]);
    expect(e.kind).toBe('distinct');
    if (e.kind === 'distinct') {
      expect(e.values).toEqual([2e-200, 1e-200]);
      sameDir(e.vectors[0], [0, 1]);
      sameDir(e.vectors[1], [1, 0]);
    }
    const f = eig2([1, 1e-200, 1e-200, 1]); // h² + bc = 1e-400: still two eigenlines, (1, 1) and (1, −1)
    if (f.kind === 'distinct') {
      sameDir(f.vectors[0], [1, 1]);
      sameDir(f.vectors[1], [1, -1]);
    } else throw new Error(f.kind);
    const g = eig2([1.5e308, 0, 0, 1e308]);
    if (g.kind === 'distinct') expect(g.values).toEqual([1.5e308, 1e308]);
    else throw new Error(g.kind);
  });
  it('#13 eig3 never invents a complex pair; symmetric matrices by Jacobi', () => {
    const d = eig3([1, 0, 0, 0, 1, 0, 0, 0, 1 + 2 ** -52]);
    expect(d.complex).toBeUndefined();
    expect(d.values).toEqual([1, 1, 1 + 2 ** -52]);
    // non-symmetric with an exactly repeated root: [[2, 1, 0], [0, 2, 0], [0, 0, 3]]
    const r = eig3([2, 1, 0, 0, 2, 0, 0, 0, 3]);
    expect(r.complex).toBeUndefined();
    expect(r.values).toEqual([2, 2, 3]);
    sameDir(r.vectors[2]!, [0, 0, 1]);
    // a rotation by 90° about z, stretched along z: 2 and ±i
    const c = eig3([0, -1, 0, 1, 0, 0, 0, 0, 2]);
    nearV(c.values, [2], 1e-15);
    near(c.complex!.re, 0, 1e-15);
    near(c.complex!.im, 1, 1e-15);
    // symmetric: A v = λ v and the vectors orthonormal
    const S: M3 = [4, 1, -2, 1, 2, 0, -2, 0, 3];
    const j = eig3(S);
    j.values.forEach((l, k) => {
      const v = j.vectors[k]!;
      const Sv = [S[0] * v[0] + S[1] * v[1] + S[2] * v[2], S[3] * v[0] + S[4] * v[1] + S[5] * v[2], S[6] * v[0] + S[7] * v[1] + S[8] * v[2]];
      nearV(Sv, v.map((x) => l * x), 1e-14);
      j.vectors.forEach((w, m) => near(v[0] * w![0] + v[1] * w![1] + v[2] * w![2], m === k ? 1 : 0, 1e-15));
    });
    near(j.values.reduce((s, x) => s + x, 0), 9, 1e-14); // trace
  });
  it('#14 inv3 is exact for the doubles given, rounded once', () => {
    const inv = inv3([1, 1 / 3, 0, 3, 1, 0, 0, 0, 1])!;
    expect(inv[8]).toBe(1);
    const det = Q.of(1).sub(Q.fromDouble(1 / 3).mul(Q.of(3)));
    expect(inv[0]).toBe(Q.of(1).div(det).toNumber());
  });
  it('#15 a line through two points with a large offset', () => {
    const f = fitLine([1e16, 1e16 + 2], [0, 2]);
    expect(f.m).toBe(1);
    expect(f.fitted).toEqual([0, 2]);
    expect(f.sse).toBe(0);
  });
});

describe('Codex re-review 2026-10-07: exact on the given numbers, and on the typed ones', () => {
  it('#11 a projection whose basis QR nearly merges: exact for the doubles given', () => {
    const r = project([1, 0, 0], [[3, 4, 5], [0.6, 0.8, 1]]);
    expect(r.independent).toBe(true);
    nearV(r.p, [5 / 9, -2 / 9, 4 / 9], 2e-16);
  });
  it('#12 eig2 with an eigenvalue beyond the double range still returns the finite one', () => {
    const e = eig2([1e308, 1e308, 1e308, 1e308]);
    expect(e.kind).toBe('distinct');
    if (e.kind === 'distinct') {
      expect(e.values).toEqual([Infinity, 0]);
      sameDir(e.vectors[0], [1, 1]);
      sameDir(e.vectors[1], [1, -1]);
    }
  });
  it('#13 eig3 keeps a tiny imaginary part, and decides repeated roots exactly', () => {
    const e = 1e-20;
    const c = eig3([1, e, 0, -e, 1, 0, 0, 0, 2]);
    expect(c.repeated).toBe(false);
    expect(c.values).toEqual([2]);
    near(c.complex!.re, 1, 1e-15);
    expect(Math.abs(c.complex!.im / e - 1)).toBeLessThan(1e-14);
    // two eigenvalues one ulp apart are distinct: both get an eigenvector
    const d = eig3([1, 1, 0, 0, 1 + 2 ** -52, 0, 0, 0, 3]);
    expect(d.repeated).toBe(false);
    expect(d.vectors.every((v) => v !== null)).toBe(true);
    // exactly repeated with one eigenline, and with a plane of eigenvectors
    const line = eig3([2, 1, 0, 0, 2, 0, 0, 0, 3]);
    expect(line.repeated).toBe(true);
    sameDir(line.vectors[0]!, [1, 0, 0]);
    expect(eig3([2, 0, 1, 0, 2, 0, 0, 0, 3]).vectors.slice(0, 2)).toEqual([null, null]);
  });
  it('#15 the intercept of a line with a large offset', () => {
    const f = fitLine([1e16, 1e16 + 2], [1e16, 1e16 + 4]);
    expect(f.m).toBe(2);
    expect(f.c).toBe(-1e16);
  });

  // what the reader typed, read exactly: decimals are decimals here, not their nearest doubles
  const typed = (...t: string[]) => t.map((x) => parseExact(x)!);
  it('typed 0.1, 0.3 | 0.3, 0.9 = 0.4, 1.2 has infinitely many solutions; its doubles have one', () => {
    const A = typed('0.1', '0.3', '0.3', '0.9') as Q2, b = typed('0.4', '1.2') as [Q, Q];
    expect(det2Q(A).isZero()).toBe(true);
    expect(solve2Q(A, b).kind).toBe('line');
    expect(solve2([0.1, 0.3, 0.3, 0.9], [0.4, 1.2]).kind).toBe('unique'); // the binary values are not 1 : 3
    expect(solve2Q(A, typed('0.4', '1.3') as [Q, Q]).kind).toBe('none');
    expect(rankQ([typed('0.1', '0.3'), typed('0.3', '0.9')])).toBe(1);
  });
  it('typed 0.1, 0.3 | 1, 3 is singular; as doubles it is not (3 × 0.1 ≠ 0.3 in binary)', () => {
    expect(det2Q(typed('0.1', '0.3', '1', '3') as Q2).isZero()).toBe(true);
    expect(det2Sign([0.1, 0.3, 1, 3])).not.toBe(0);
    expect(det2Q(typed('1/3', '1/7', '7', '3') as Q2).isZero()).toBe(true); // fractions are exact too
  });
  it('a repeated eigenvalue in the typed decimals (2×2 and 3×3)', () => {
    // [[0.1, 0.2], [−0.05, 0.3]]: ((a − d)/2)² + bc = 0.01 − 0.01 = 0 → one eigenvalue 0.2, one eigenline
    const A = typed('0.1', '0.2', '-0.05', '0.3') as Q2;
    const e = eig2Q(A);
    expect(e.kind).toBe('defective');
    if (e.kind === 'defective') {
      expect(e.value).toBe(0.2);
      sameDir(e.vector, [1, 0.5]);
    }
    expect(eig2([0.1, 0.2, -0.05, 0.3]).kind).not.toBe('defective');
    const M = typed('0.1', '0.2', '0', '-0.05', '0.3', '0', '0', '0', '1') as Q3;
    const t = eig3Q(M);
    expect(t.repeated).toBe(true);
    expect(t.values).toEqual([0.2, 0.2, 1]);
    expect(eig3([0.1, 0.2, 0, -0.05, 0.3, 0, 0, 0, 1]).repeated).toBe(false);
    expect(det3Q(typed('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9') as Q3).isZero()).toBe(true);
    expect(det3Exact([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).isZero()).toBe(false);
    expect(solve3Q(typed('1', '2', '3', '4', '5', '6', '7', '8', '9') as Q3, typed('0.1', '0.4', '0.7') as [Q, Q, Q]).kind).toBe('line');
  });
});

describe('Codex third review 2026-10-07: scales, exact null spaces, tiny imaginary parts', () => {
  it('eig2 keeps a representable small eigenvalue and a vector component next to huge ones', () => {
    const e = eig2([1e308, 0, 0, 1e-308]);
    expect(e.kind === 'distinct' && e.values).toEqual([1e308, 1e-308]);
    const d = eig2([1e308, 1e-308, 0, 1e308]);
    expect(d.kind).toBe('defective');
    if (d.kind === 'defective') expect(d.vector).toEqual([1, 0]);
  });
  it('a symmetric matrix with an exactly repeated root: its vectors come from the exact entries', () => {
    // M = I + ε·uuᵀ/5, u = (1, 2, 0), ε = 23e-17: eigenvalues 1, 1 and 1 + ε on (1, 2, 0)/√5
    const eps = Q.of(23n, 10n ** 17n), f = (a: number) => eps.mul(Q.of(a, 5));
    const M = [Q.ONE.add(f(1)), f(2), Q.ZERO, f(2), Q.ONE.add(f(4)), Q.ZERO, Q.ZERO, Q.ZERO, Q.ONE] as Q3;
    const r = eig3Q(M);
    expect(r.repeated).toBe(true);
    expect(r.values).toEqual([1, 1, 1 + 23e-17]);
    sameDir(r.vectors[2]!, [1, 2, 0]);
    // the repeated pair: an orthonormal basis of the plane orthogonal to it
    const [v0, v1, v2] = r.vectors as V3[];
    for (const [x, y, want] of [[v0, v1, 0], [v0, v2, 0], [v1, v2, 0], [v0, v0, 1], [v1, v1, 1]] as [V3, V3, number][])
      near(x[0] * y[0] + x[1] * y[1] + x[2] * y[2], want, 1e-15);
    const triple = eig3([2, 1, 0, 0, 2, 1, 0, 0, 2]); // one eigenline for a triple root
    expect(triple.vectors).toEqual([[1, 0, 0], [1, 0, 0], [1, 0, 0]]);
  });
  it('imaginary parts at any scale: ±1e-200 i, ±i next to 1e78, ±1e-20 i next to 1e78', () => {
    const B = (e: number, g: number) => eig3([0, e, 0, -e, 0, 0, 0, 0, g]);
    const a = B(1e-200, 0);
    expect(a.complex!.im).toBe(1e-200);
    expect(a.values).toEqual([0]);
    sameDir(a.vectors[0]!, [0, 0, 1]);
    expect(Math.abs(B(1, 1e78).complex!.im - 1)).toBeLessThan(1e-15);
    expect(Math.abs(B(1e-20, 1e78).complex!.im / 1e-20 - 1)).toBeLessThan(1e-14);
  });
  it('a line through tiny data: slope 1, intercept 0, no exception', () => {
    const f = fitLine([0, 1e-200], [0, 1e-200]);
    expect(f.m).toBe(1);
    expect(f.c).toBe(0);
  });
});

describe('exact inverse and solutions (F08, 2026-10-07)', () => {
  const Qs = (...xs: string[]) => xs.map((x) => parseExact(x)!) as Q2;
  it('inv2Q: ⅓[[2, 1], [−1, 1]] for [[1, −1], [1, 2]]; null when det is exactly 0; 1e13 for [[1, 1], [0, 1e-13]]', () => {
    expect(inv2Q(Qs('1', '-1', '1', '2'))!.map(String)).toEqual(['2/3', '1/3', '-1/3', '1/3']);
    expect(inv2Q(Qs('1', '1/3', '3', '1'))).toBeNull();
    expect(inv2Q([Q.ONE, Q.ONE, Q.ZERO, Q.of(1, 10n ** 13n)])![3].toNumber()).toBe(1e13);
  });
  it('solve2Exact: the unique solution as rationals; a line as one equation; none; the plane', () => {
    const u = solve2Exact(Qs('1', '-1', '1', '2'), [parseExact('1')!, parseExact('4')!]);
    expect(u.kind === 'unique' && u.x.map(String)).toEqual(['2', '1']);
    const third = solve2Exact(Qs('3', '0', '0', '7'), [Q.ONE, Q.ONE]);
    expect(third.kind === 'unique' && third.x.map(String)).toEqual(['1/3', '1/7']);
    const l = solve2Exact(Qs('2', '1', '1', '0.5'), [parseExact('3')!, parseExact('1.5')!]);
    expect(l.kind === 'line' && [...l.row, l.rhs].map(String)).toEqual(['2', '1', '3']);
    const bottom = solve2Exact(Qs('0', '0', '1', '2'), [Q.ZERO, parseExact('4')!]);
    expect(bottom.kind === 'line' && [...bottom.row, bottom.rhs].map(String)).toEqual(['1', '2', '4']);
    expect(solve2Exact(Qs('2', '1', '1', '0.5'), [Q.ZERO, Q.ONE]).kind).toBe('none');
    expect(solve2Exact(Qs('0', '0', '0', '0'), [Q.ZERO, Q.ZERO]).kind).toBe('plane');
  });
});
