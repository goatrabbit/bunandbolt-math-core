// Example 2: a singular matrix stays singular.
//
// The rows of [[0.1, 0.7], [0.3, 2.1]] are proportional (the second is three times the first), so the matrix is
// singular. As doubles, though, the entries are slightly different numbers (0.1 is 3602879701896397 / 2⁵⁵), and those
// numbers do not have proportional rows: solve2 decides exactly on the doubles it is given, so it finds a determinant
// of about 2.8e-17 and a "unique" solution with a condition number near 10¹⁷. Read as the decimals the reader typed,
// the determinant is exactly 0 and the system gets its true case: a line of solutions, or none. linalg.ts never rounds the other way either: a
// nearly singular matrix gets its true, possibly enormous, answer and a condition number, and a page that wants a
// tolerance applies it itself.
import { describe, expect, it } from 'vitest';
import { parseExact, type Q } from '../src/lib/math/rational';
import { det2, det2Q, rankQ, solve2, solve2Q, type M2, type Q2 } from '../src/lib/math/linalg';

const typed = ['0.1', '0.7', '0.3', '2.1'];
const exact = (s: string): Q => parseExact(s)!;

describe('[[0.1, 0.7], [0.3, 2.1]]', () => {
  it('is not singular as doubles', () => {
    const A = typed.map(Number) as M2;
    expect(det2(A)).not.toBe(0);
    const s = solve2(A, [1, 3]);
    expect(s.kind).toBe('unique');
    if (s.kind === 'unique') expect(s.condition).toBeGreaterThan(1e16);
  });

  it('is singular as typed', () => {
    const A = typed.map(exact) as Q2;
    expect(det2Q(A).isZero()).toBe(true);
    expect(rankQ([[A[0], A[1]], [A[2], A[3]]])).toBe(1);
    expect(solve2Q(A, [exact('1'), exact('3')]).kind).toBe('line');
    expect(solve2Q(A, [exact('1'), exact('2')]).kind).toBe('none');
  });
});
