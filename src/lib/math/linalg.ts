// Small linear algebra for the linear-algebra, phase-portrait and least-squares pages: 2×2 and 3×3 determinants,
// inverses, solves and eigen-decompositions, orthogonal projection onto a span, and least squares; and 4 × 4 plane
// rotations for the tesseract page (at the end).
//
// Exactness policy (direction drafts, Codex reviews): a matrix is reported singular, an eigenvalue repeated, a basis
// degenerate, only when that is exactly so (determinant exactly 0, discriminant exactly 0, a vector exactly in the
// span). Nearly-singular input gets its true, possibly enormous, answer plus a `condition` number for the page to warn
// with — it is never rounded into a different case. Callers that need a tolerance (snapping, a declared preset) apply
// it themselves.
//
// Decisions from typed values go through Q, from the typed text; numbers are only for drawing. A reader who types
// 0.1, 0.3 | 0.3, 0.9 = 0.4, 1.2 means decimals: read each entry with `parseExact` (rational.ts: "0.1" → 1/10,
// "1/3" → 1/3) and call the Q entry points — det2Q, det3Q, rankQ, rrefQ, solve2Q, solve3Q, eig2Q, eig3Q — and the
// answer is "infinitely many", as the reader means. The number versions (det2Sign, solve2, eig2, …) take the doubles
// a page computes or drags and decide exactly on their binary values (the double 0.1 is not 1/10, so the same system
// passed as doubles has one solution); use them for values that never were decimals. Every function returns numbers
// for drawing, rounded once from the exact result.
//
// Matrices are row-major arrays: M2 = [a, b, c, d] is [[a, b], [c, d]]; M3 has nine entries; Q2 / Q3 the same in Q.

import { Q, scale2 } from './rational';

export type V2 = [number, number];
export type V3 = [number, number, number];
export type M2 = [number, number, number, number];
export type M3 = [number, number, number, number, number, number, number, number, number];

const ex = Q.fromDouble;
const qv = (A: readonly number[]) => A.map(ex);

/** A 2×2 or 3×3 of exact rationals, row-major like M2 / M3 — typically the typed entries, read with `parseExact`. */
export type Q2 = [Q, Q, Q, Q];
export type Q3 = [Q, Q, Q, Q, Q, Q, Q, Q, Q];

/** det A, exactly. */
export const det2Q = (A: Q2): Q => A[0].mul(A[3]).sub(A[1].mul(A[2]));
/** Exact sign of det A for the doubles given. */
export const det2Sign = (A: M2): -1 | 0 | 1 => det2Q(qv(A) as Q2).sign();

/** det M, exactly. */
export function det3Q(m: Q3): Q {
  return m[0].mul(m[4].mul(m[8]).sub(m[5].mul(m[7])))
    .sub(m[1].mul(m[3].mul(m[8]).sub(m[5].mul(m[6]))))
    .add(m[2].mul(m[3].mul(m[7]).sub(m[4].mul(m[6]))));
}
/** Exact determinant of a 3×3 of doubles, as a rational. */
export const det3Exact = (M: M3): Q => det3Q(qv(M) as Q3);

/**
 * Exact row reduction of a matrix of doubles (rows × cols) over the rationals: the pivot columns, in order, and the
 * reduced rows. Rank questions are answered exactly on the numbers given.
 */
export function rref(rows: number[][]): { R: Q[][]; pivots: number[] } {
  return rrefQ(rows.map((r) => r.map(ex)));
}

/** The same for a rational matrix (the argument is not changed). */
export function rrefQ(rows: Q[][]): { R: Q[][]; pivots: number[] } {
  const R = rows.map((r) => r.slice());
  const m = R.length, n = m ? R[0].length : 0;
  const pivots: number[] = [];
  let row = 0;
  for (let col = 0; col < n && row < m; col++) {
    let p = -1;
    for (let r = row; r < m; r++) if (!R[r][col].isZero()) { p = r; break; }
    if (p < 0) continue;
    [R[row], R[p]] = [R[p], R[row]];
    const pv = R[row][col];
    R[row] = R[row].map((v) => v.div(pv));
    for (let r = 0; r < m; r++) {
      if (r === row || R[r][col].isZero()) continue;
      const f = R[r][col];
      R[r] = R[r].map((v, k) => v.sub(f.mul(R[row][k])));
    }
    pivots.push(col);
    row++;
  }
  return { R, pivots };
}

/** The rank of a rational matrix, exactly (are these typed vectors independent? is det 0?). */
export const rankQ = (rows: Q[][]): number => rrefQ(rows).pivots.length;

/** A unit vector along v without overflow (v scaled by its largest component first); v itself when it is 0. */
function unit<T extends number[]>(v: T): T {
  const m = Math.max(...v.map(Math.abs));
  if (m === 0 || !Number.isFinite(m)) return v;
  const w = v.map((x) => x / m), n = Math.hypot(...w);
  return w.map((x) => x / n) as T;
}

/** A power of two near the largest entry (exponent k), so that A·2⁻ᵏ has entries of size about 1 — exact scaling. */
function scaleExponent(A: Q[]): number {
  const k = Math.max(...A.map((q) => q.ilog2()));
  return Number.isFinite(k) ? k : 0;
}
const timesPow2 = (q: Q, k: number) => (k >= 0 ? q.mul(Q.of(1n << BigInt(k))) : q.div(Q.of(1n << BigInt(-k))));

/** A unit vector from exact components: divided by the largest of them first, so a tiny one does not vanish against a
 *  huge one before the rounding that would remove it anyway. */
function unitQ(v: Q[]): number[] {
  const m = v.reduce((x, q) => Q.max(x, q.abs()), Q.ZERO);
  return m.isZero() ? v.map(() => 0) : unit(v.map((q) => q.div(m).toNumber()));
}

// ---------------------------------------------------------------- 2×2

export const det2 = (A: M2) => A[0] * A[3] - A[1] * A[2];
export const trace2 = (A: M2) => A[0] + A[3];
export const mul2 = (A: M2, B: M2): M2 => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
export const apply2 = (A: M2, v: V2): V2 => [A[0] * v[0] + A[1] * v[1], A[2] * v[0] + A[3] * v[1]];

/** Columns of a 2×2: Aî and Aĵ. */
export const columns2 = (A: M2): [V2, V2] => [[A[0], A[2]], [A[1], A[3]]];

/** ∞-norm condition number ‖A‖·‖A⁻¹‖, from exact values (Infinity when det A = 0). */
export function cond2Q(A: Q2): number {
  const D = det2Q(A);
  if (D.isZero()) return Infinity;
  const [a, b, c, d] = A.map((q) => q.abs());
  return Q.max(a.add(b), c.add(d)).mul(Q.max(d.add(b), c.add(a))).div(D.abs()).toNumber();
}
export const cond2 = (A: M2): number => cond2Q(qv(A) as Q2);

/** A⁻¹, or null when det A is exactly 0. */
export function inv2(A: M2): M2 | null {
  if (det2Sign(A) === 0) return null;
  const d = ex(A[0]).mul(ex(A[3])).sub(ex(A[1]).mul(ex(A[2])));
  return [ex(A[3]).div(d).toNumber(), ex(A[1]).neg().div(d).toNumber(), ex(A[2]).neg().div(d).toNumber(), ex(A[0]).div(d).toNumber()];
}

export type Solve2 =
  | { kind: 'unique'; x: V2; condition: number }
  | { kind: 'none' }
  | { kind: 'line'; point: V2; direction: V2 }
  | { kind: 'plane' };

/**
 * Solve A x = b exactly by case: unique (Cramer, with the condition number), none, a line of solutions, or every x
 * (A = 0 and b = 0). The case is decided exactly on the rationals given; the numbers returned are for drawing.
 */
export function solve2Q(A: Q2, b: [Q, Q]): Solve2 {
  const [a, bb, c, d] = A, [p, q] = b;
  const D = det2Q(A);
  if (!D.isZero()) {
    return { kind: 'unique', x: [p.mul(d).sub(bb.mul(q)).div(D).toNumber(), a.mul(q).sub(p.mul(c)).div(D).toNumber()], condition: cond2Q(A) };
  }
  const { R, pivots } = rrefQ([[a, bb, p], [c, d, q]]);
  if (pivots.includes(2)) return { kind: 'none' };
  if (pivots.length === 0) return { kind: 'plane' };
  // rank 1: R[0] = [1, k, r] or [0, 1, r]
  if (pivots[0] === 0) return { kind: 'line', point: [R[0][2].toNumber(), 0], direction: unit([-R[0][1].toNumber(), 1] as V2) };
  return { kind: 'line', point: [0, R[0][2].toNumber()], direction: [1, 0] };
}
/** The same for the doubles a page holds (decided exactly on those binary values; see the header). */
export const solve2 = (A: M2, b: V2): Solve2 => solve2Q(qv(A) as Q2, qv(b) as [Q, Q]);

/** A⁻¹ exactly ((1/det A)·[[d, −b], [−c, a]]), or null when det A is exactly 0. */
export function inv2Q(A: Q2): Q2 | null {
  const D = det2Q(A);
  if (D.isZero()) return null;
  return [A[3].div(D), A[1].neg().div(D), A[2].neg().div(D), A[0].div(D)];
}

export type Solve2Exact =
  | { kind: 'unique'; x: [Q, Q] }
  /** every x on the line row·x = rhs (a non-zero row of A and its entry of b), parallel to the kernel */
  | { kind: 'line'; row: [Q, Q]; rhs: Q; point: V2; direction: V2 }
  | { kind: 'none' }
  | { kind: 'plane' };

/**
 * A x = b with the answer kept exact (solve2Q decides the case; this also returns the solution as rationals, for a
 * page that shows it or builds on it — F08's preimage, F03's intersection): the unique x by Cramer's rule, or the
 * line of solutions as one equation row·x = rhs taken from a non-zero row of A. Numbers (point, direction) are for
 * drawing only.
 */
export function solve2Exact(A: Q2, b: [Q, Q]): Solve2Exact {
  const s = solve2Q(A, b);
  if (s.kind === 'none' || s.kind === 'plane') return { kind: s.kind };
  if (s.kind === 'unique') {
    const D = det2Q(A);
    return { kind: 'unique', x: [b[0].mul(A[3]).sub(A[1].mul(b[1])).div(D), A[0].mul(b[1]).sub(b[0].mul(A[2])).div(D)] };
  }
  const top = !A[0].isZero() || !A[1].isZero();
  return { kind: 'line', row: top ? [A[0], A[1]] : [A[2], A[3]], rhs: top ? b[0] : b[1], point: s.point, direction: s.direction };
}

export type Eigen2 =
  | { kind: 'distinct'; values: [number, number]; vectors: [V2, V2] }
  | { kind: 'complex'; re: number; im: number }
  | { kind: 'scalar'; value: number } // A = λI: every direction is an eigenvector
  | { kind: 'defective'; value: number; vector: V2 }; // one eigenline only (e.g. a shear)

/**
 * Eigenvalues and unit eigenvectors of a real 2×2. `kind` follows the exact sign of the discriminant
 * ((a − d)/2)² + bc. The matrix is first scaled by a power of two (exactly) to entries near 1, so nothing over- or
 * underflows on the way; an eigenvalue beyond the double range comes back as ±Infinity, the other stays finite.
 * Eigenvalues are ordered λ₁ ≥ λ₂ (real case).
 */
export function eig2Q(A0: Q2): Eigen2 {
  const k = scaleExponent(A0);
  const A = A0.map((q) => timesPow2(q, -k)) as Q2;
  const hq = A[0].sub(A[3]).div(Q.of(2)), sq = A[0].add(A[3]).div(Q.of(2));
  const discQ = hq.mul(hq).add(A[1].mul(A[2]));
  const value = (q: Q) => timesPow2(q, k).toNumber(); // back to the original scale, rounded once
  if (discQ.sign() < 0) return { kind: 'complex', re: value(sq), im: scale2(discQ.neg().sqrtToNumber(), k) };
  if (discQ.sign() === 0) {
    if (A[1].isZero() && A[2].isZero()) return { kind: 'scalar', value: value(sq) };
    // A − sI = [[h, b], [c, −h]] has null vectors (b, −h) and (h, c): take the longer
    const vector = A[1].abs().cmp(A[2].abs()) >= 0 ? unitQ([A[1], hq.neg()]) : unitQ([hq, A[2]]);
    return { kind: 'defective', value: value(sq), vector: vector as V2 };
  }
  const rQ = ex(discQ.sqrtToNumber());
  // λ = s ± r: the larger-magnitude one as s ± r, the other as det A / λ₁ — both kept rational until the final rounding,
  // so a small eigenvalue next to a large one keeps its digits and neither under- nor overflows on the way
  const bigQ = sq.sign() >= 0 ? sq.add(rQ) : sq.sub(rQ);
  const smallQ = det2Q(A).div(bigQ);
  const [l1, l2] = bigQ.cmp(smallQ) >= 0 ? [bigQ, smallQ] : [smallQ, bigQ];
  // eigenvectors from h = (a − d)/2 and r, never from A − λI with a rounded λ: for s + r, (r + h, c) or (b, r − h); for
  // s − r, (b, −(r + h)) or (r − h, −c) — each pair picks the form whose sum does not cancel
  const vPlus = hq.sign() >= 0 ? unitQ([rQ.add(hq), A[2]]) : unitQ([A[1], rQ.sub(hq)]);
  const vMinus = hq.sign() >= 0 ? unitQ([A[1], rQ.add(hq).neg()]) : unitQ([rQ.sub(hq), A[2].neg()]);
  return { kind: 'distinct', values: [value(l1), value(l2)], vectors: [vPlus as V2, vMinus as V2] };
}
/** The same for the doubles a page holds. */
export const eig2 = (A: M2): Eigen2 => eig2Q(qv(A) as Q2);

// ---------------------------------------------------------------- 3×3

export const det3 = (M: M3) =>
  M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
export const apply3 = (M: M3, v: V3): V3 => [
  M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
  M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
  M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
];
export const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a: V3) => Math.hypot(a[0], a[1], a[2]);

/** The exact inverse, or null when det M = 0. */
function inv3Q(M: Q3): Q3 | null {
  const D = det3Q(M);
  if (D.isZero()) return null;
  const [a, b, c, e, f, g, h, i, j] = M;
  const cof = (p: Q, q: Q, r: Q, t: Q) => p.mul(q).sub(r.mul(t)).div(D);
  return [
    cof(f, j, g, i), cof(c, i, b, j), cof(b, g, c, f),
    cof(g, h, e, j), cof(a, j, c, h), cof(c, e, a, g),
    cof(e, i, f, h), cof(b, h, a, i), cof(a, f, b, e),
  ];
}

/** The inverse, each entry exact for the doubles given and rounded once (cofactors in floating point can cancel to 0). */
export function inv3(M: M3): M3 | null {
  const I = inv3Q(qv(M) as Q3);
  return I && (I.map((q) => q.toNumber()) as M3);
}

/** ∞-norm condition number, from exact values. */
export function cond3Q(M: Q3): number {
  const I = inv3Q(M);
  if (!I) return Infinity;
  const nrm = (A: Q[]) => [0, 3, 6].map((r) => A[r].abs().add(A[r + 1].abs()).add(A[r + 2].abs())).reduce((x, y) => Q.max(x, y));
  return nrm(M).mul(nrm(I)).toNumber();
}
export const cond3 = (M: M3): number => cond3Q(qv(M) as Q3);

export type Solve3 =
  | { kind: 'unique'; x: V3; condition: number }
  | { kind: 'none'; rank: number }
  | { kind: 'line'; point: V3; direction: V3 }
  | { kind: 'plane'; point: V3; normal: V3 }
  | { kind: 'space' };

/**
 * Solve A x = b (3×3) by exact row reduction, rounded once at the end: the case (one solution, none, a line, a plane,
 * all of space) is exact on the rationals given, and a nearly singular system still gets its true solution together
 * with its condition number.
 */
export function solve3Q(A: Q3, b: [Q, Q, Q]): Solve3 {
  const { R, pivots } = rrefQ([
    [A[0], A[1], A[2], b[0]],
    [A[3], A[4], A[5], b[1]],
    [A[6], A[7], A[8], b[2]],
  ]);
  if (pivots.includes(3)) return { kind: 'none', rank: pivots.length - 1 };
  const rank = pivots.length;
  if (rank === 3) return { kind: 'unique', x: [R[0][3].toNumber(), R[1][3].toNumber(), R[2][3].toNumber()], condition: cond3Q(A) };
  if (rank === 0) return { kind: 'space' };
  const point: V3 = [0, 0, 0];
  pivots.forEach((c, r) => (point[c] = R[r][3].toNumber()));
  const free = [0, 1, 2].filter((c) => !pivots.includes(c));
  if (rank === 2) {
    const f = free[0];
    const dir: V3 = [0, 0, 0];
    dir[f] = 1;
    pivots.forEach((c, r) => (dir[c] = -R[r][f].toNumber()));
    return { kind: 'line', point, direction: dir };
  }
  return { kind: 'plane', point, normal: [R[0][0].toNumber(), R[0][1].toNumber(), R[0][2].toNumber()] };
}
/** The same for the doubles a page holds. */
export const solve3 = (A: M3, b: V3): Solve3 => solve3Q(qv(A) as Q3, qv(b) as [Q, Q, Q]);

/** Real roots of t³ + p t² + q t + r = 0 (sorted), or one real root and a complex pair. Trigonometric / Cardano. */
export function cubicRoots(p: number, q: number, r: number): { real: number[]; complex?: { re: number; im: number } } {
  // depressed cubic x³ + Px + Q with t = x − p/3
  const P = q - (p * p) / 3, Q = (2 * p * p * p) / 27 - (p * q) / 3 + r, shift = -p / 3;
  const D = (Q * Q) / 4 + (P * P * P) / 27;
  if (P === 0 && Q === 0) return { real: [shift, shift, shift] };
  if (D > 0) {
    const s = Math.sqrt(D);
    const u = Math.cbrt(-Q / 2 + s), v = Math.cbrt(-Q / 2 - s);
    const x = u + v;
    return { real: [x + shift], complex: { re: -x / 2 + shift, im: (Math.sqrt(3) / 2) * Math.abs(u - v) } };
  }
  if (D === 0) {
    const u = Math.cbrt(-Q / 2);
    return { real: [2 * u + shift, -u + shift, -u + shift].sort((a, b) => a - b) };
  }
  const m = 2 * Math.sqrt(-P / 3);
  const theta = Math.acos(Math.max(-1, Math.min(1, (3 * Q) / (P * m)))) / 3;
  const roots = [0, 1, 2].map((k) => m * Math.cos(theta - (2 * Math.PI * k) / 3) + shift);
  return { real: roots.sort((a, b) => a - b) };
}

/** Symmetric 3×3 by cyclic Jacobi rotations (Golub & Van Loan §8.5): real eigenvalues, orthonormal eigenvectors. */
function jacobi3(M: M3): { values: number[]; vectors: V3[] } {
  const A = [[M[0], M[1], M[2]], [M[3], M[4], M[5]], [M[6], M[7], M[8]]];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = A[p][q];
      if (apq === 0) continue;
      // negligible next to both diagonal entries: set to zero (NR §11.1)
      if (Math.abs(apq) <= 1e-18 * Math.abs(A[p][p]) && Math.abs(apq) <= 1e-18 * Math.abs(A[q][q])) {
        A[p][q] = A[q][p] = 0;
        continue;
      }
      rotated = true;
      const theta = (A[q][q] - A[p][p]) / (2 * apq);
      const t = Math.abs(theta) > 1e150 ? 1 / (2 * theta) : (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      A[p][p] -= t * apq;
      A[q][q] += t * apq;
      A[p][q] = A[q][p] = 0;
      for (let r = 0; r < 3; r++) {
        if (r !== p && r !== q) {
          const arp = A[r][p], arq = A[r][q];
          A[r][p] = A[p][r] = c * arp - sn * arq;
          A[r][q] = A[q][r] = sn * arp + c * arq;
        }
        const vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - sn * vrq;
        V[r][q] = sn * vrp + c * vrq;
      }
    }
    if (!rotated) break;
  }
  const order = [0, 1, 2].sort((i, j) => A[i][i] - A[j][j]);
  return { values: order.map((k) => A[k][k]), vectors: order.map((k) => [V[0][k], V[1][k], V[2][k]] as V3) };
}

export interface Eigen3 {
  /** real eigenvalues, ascending (all three, or the one real root of a complex pair) */
  values: number[];
  /** a unit eigenvector per value; null for a repeated root whose eigenvectors fill a plane (or all of space) */
  vectors: (V3 | null)[];
  /** the complex pair re ± i·im, when there is one */
  complex?: { re: number; im: number };
  /** the characteristic polynomial has a repeated root — decided exactly */
  repeated: boolean;
}

/**
 * Eigenvalues of a real 3×3, with the case — three distinct real roots, a repeated one, or a complex pair — decided by
 * the exact sign of the characteristic polynomial's discriminant, so rounding never invents a complex pair or a
 * repeated root, nor loses one.
 * - Repeated roots are exact rationals (−3R/2P and 3R/P of the depressed cubic x³ + Px + R, or tr/3 when P = R = 0);
 *   their eigenvectors come from the exact null space of M − λI (one line, or null for a plane).
 * - Symmetric (exactly): Jacobi rotations give the vectors, always an orthonormal set.
 * - Otherwise simple real roots by the trigonometric form, polished by Newton; one real root and a complex pair by
 *   Cardano, the pair's imaginary part from the exact discriminant Δ = −4·im²·|λ − z|⁴, so a pair 1 ± 1e-20 i keeps it.
 */
export function eig3Q(M0: Q3): Eigen3 {
  // scaled by a power of two (exact) to entries near 1: nothing over- or underflows on the way; values scale back
  const k = scaleExponent(M0);
  const M = M0.map((q) => timesPow2(q, -k)) as Q3;
  const back = (x: number) => scale2(x, k);
  const sym = M[1].eq(M[3]) && M[2].eq(M[6]) && M[5].eq(M[7]);
  // λ³ − tr λ² + m λ − det = 0; depressed with λ = x + tr/3: x³ + P x + R = 0, P = m − tr²/3, R = −2tr³/27 + tr·m/3 − det
  const tr = M[0].add(M[4]).add(M[8]);
  const m = M[0].mul(M[4]).sub(M[1].mul(M[3])).add(M[0].mul(M[8]).sub(M[2].mul(M[6]))).add(M[4].mul(M[8]).sub(M[5].mul(M[7])));
  const det = det3Q(M);
  const third = Q.of(1, 3);
  const P = m.sub(tr.mul(tr).mul(third));
  const R = tr.pow(3).mul(Q.of(-2, 27)).add(tr.mul(m).mul(third)).sub(det);
  const disc = P.pow(3).mul(Q.of(4)).add(R.mul(R).mul(Q.of(27))).neg(); // −(4P³ + 27R²): the discriminant
  const shift = tr.mul(third);
  const num = M.map((q) => q.toNumber()) as M3;
  // exact repeated roots, ascending
  let exact: Q[] | null = null;
  if (P.isZero() && R.isZero()) exact = [shift, shift, shift];
  else if (disc.isZero()) {
    const double = R.mul(Q.of(-3)).div(P.mul(Q.of(2))).add(shift), single = R.mul(Q.of(3)).div(P).add(shift);
    exact = double.cmp(single) < 0 ? [double, double, single] : [single, double, double];
  }
  if (exact) {
    const values = exact.map((q) => timesPow2(q, k).toNumber());
    // eigenvectors from the exact null space of M − λI (computing them from the rounded matrix would lose a direction
    // that only the exact entries carry)
    const vectors: (V3 | null)[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0 && exact[i].eq(exact[i - 1])) continue; // done with its equal neighbour
      const basis = nullSpace3(M, exact[i]).map((v) => unitQ(v) as V3);
      const mult = exact.filter((q) => q.eq(exact![i])).length;
      if (basis.length === 1) for (let j = 0; j < mult; j++) vectors.push(basis[0]); // one eigenline, however repeated
      else if (!sym) for (let j = 0; j < mult; j++) vectors.push(null); // a plane (or all of space) of eigenvectors
      else vectors.push(...orthonormal(basis).slice(0, mult)); // symmetric: an orthonormal basis of that plane
    }
    return { values, vectors, repeated: true };
  }
  if (sym) {
    const J = jacobi3(num);
    return { values: J.values.map(back), vectors: J.vectors, repeated: false };
  }
  // simple roots, polished by Newton on λ³ − tr λ² + m λ − det (coefficients rounded once from the exact values)
  const c2 = tr.toNumber(), c1 = m.toNumber(), c0 = det.toNumber();
  const polish = (l: number) => {
    for (let it = 0; it < 3; it++) {
      const f = ((l - c2) * l + c1) * l - c0, df = (3 * l - 2 * c2) * l + c1;
      if (df === 0 || f === 0) break;
      const next = l - f / df;
      if (!(Math.abs(((next - c2) * next + c1) * next - c0) < Math.abs(f))) break;
      l = next;
    }
    return l;
  };
  let values: number[], complex: { re: number; im: number } | undefined;
  if (disc.sign() > 0) {
    values = trigRoots(P.toNumber(), R.toNumber(), shift.toNumber()).map(polish).sort((x, y) => x - y); // P < 0 here
  } else {
    // one real root and a complex pair (Cardano), with √(R²/4 + P³/27) = √(−disc/108) from the exact value
    const sq = disc.neg().div(Q.of(108)).sqrtToNumber(), half = -R.toNumber() / 2, sh = shift.toNumber();
    const u = Math.cbrt(half + sq), v = Math.cbrt(half - sq);
    const l = polish(u + v + sh);
    values = [l];
    const re = (c2 - l) / 2;
    complex = { re: back(re), im: back(pairImaginary(disc.neg().sqrtToNumber() / 2, l - re)) };
  }
  const [a, b, c, d, e, f, g, h, i] = num;
  const vectors = values.map((l) => {
    // null vector of M − λI for a simple root: the longest cross product of two rows
    const r0: V3 = [a - l, b, c], r1: V3 = [d, e - l, f], r2: V3 = [g, h, i - l];
    const cands = [cross3(r0, r1), cross3(r0, r2), cross3(r1, r2)];
    let best = cands[0];
    for (const v of cands) if (norm3(v) > norm3(best)) best = v;
    return norm3(best) === 0 ? null : unit(best);
  });
  values = values.map(back);
  return complex ? { values, vectors, complex, repeated: false } : { values, vectors, repeated: false };
}

/** A basis of the null space of M − λI over Q (one vector per free column). */
function nullSpace3(M: Q3, l: Q): Q[][] {
  const { R, pivots } = rrefQ([0, 1, 2].map((r) => [0, 1, 2].map((c) => (c === r ? M[3 * r + c].sub(l) : M[3 * r + c]))));
  return [0, 1, 2].filter((c) => !pivots.includes(c)).map((f) => {
    const v = [Q.ZERO, Q.ZERO, Q.ZERO];
    v[f] = Q.ONE;
    pivots.forEach((c, r) => (v[c] = R[r][f].neg()));
    return v;
  });
}

/** Gram–Schmidt (twice) on unit vectors that span a plane or all of space. */
function orthonormal(vs: V3[]): V3[] {
  const out: V3[] = [];
  for (const v of vs) {
    let w = v.slice() as V3;
    for (let pass = 0; pass < 2; pass++) for (const u of out) {
      const d = dot3(u, w);
      w = [w[0] - d * u[0], w[1] - d * u[1], w[2] - d * u[2]];
    }
    out.push(unit(w));
  }
  return out;
}
/** The same for the doubles a page holds. */
export const eig3 = (M: M3): Eigen3 => eig3Q(qv(M) as Q3);

/**
 * The imaginary part of a cubic's complex pair re ± i·im from its exact discriminant: Δ = −4·im²·((λ − re)² + im²)²
 * (λ the real root), so v = im solves v³ + A·v = S with A = (λ − re)² and S = √(−Δ)/2 — taken from the exact Δ, so it
 * neither under- nor overflows. The left side is increasing and convex in v ≥ 0: Newton from the upper bound
 * min(S/A, ∛S) descends to the root.
 */
function pairImaginary(S: number, gap: number): number {
  const A = gap * gap;
  let v = Math.min(A > 0 ? S / A : Infinity, Math.cbrt(S));
  for (let it = 0; it < 200; it++) {
    const f = v * v * v + A * v - S, df = 3 * v * v + A;
    if (!(f > 0) || !(df > 0)) break;
    const next = v - f / df;
    if (!(next < v)) break;
    v = Math.max(0, next);
  }
  return v;
}

/** The three real roots of x³ + Px + R (P < 0), shifted, by the trigonometric form with the cosine argument clamped. */
function trigRoots(P: number, R: number, shift: number): number[] {
  const m = 2 * Math.sqrt(-P / 3);
  const theta = Math.acos(Math.max(-1, Math.min(1, (3 * R) / (P * m)))) / 3;
  return [0, 1, 2].map((k) => m * Math.cos(theta - (2 * Math.PI * k) / 3) + shift).sort((x, y) => x - y);
}

// ---------------------------------------------------------------- projection and least squares

export interface Projection {
  /** the projection p of b onto span(basis) */
  p: number[];
  /** the error e = b − p (orthogonal to every basis vector) */
  e: number[];
  /** coordinates c with p = Σ cᵢ basisᵢ, solved through QR (not through the normal equations) */
  coords: number[];
  /** false when the basis vectors are exactly dependent (coords then meaningless: p is still right) */
  independent: boolean;
}

/** Thin QR of the columns by modified Gram–Schmidt with one re-orthogonalisation pass; columns not in `keep` (exactly
 *  dependent on earlier ones) get a zero direction so indices stay aligned. */
function qr(cols: number[][], keep: Set<number>): { Q: number[][]; R: number[][]; rank: number } {
  const n = cols.length;
  const Q: number[][] = [], R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  let rank = 0;
  for (let j = 0; j < n; j++) {
    const v = cols[j].slice();
    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k < Q.length; k++) {
        const r = Q[k].reduce((s, q, i) => s + q * v[i], 0);
        R[k][j] += r;
        for (let i = 0; i < v.length; i++) v[i] -= r * Q[k][i];
      }
    }
    const nv = Math.hypot(...v);
    if (!keep.has(j) || nv === 0) {
      Q.push(new Array(v.length).fill(0)); // dependent column: a zero direction keeps indices aligned
      R[j][j] = 0;
      continue;
    }
    R[j][j] = nv;
    Q.push(v.map((x) => x / nv));
    rank++;
  }
  return { Q, R, rank };
}

/** Up to this many numbers (dimension × basis vectors) a projection is computed exactly: 100 points with three basis
 *  vectors take about 5 ms. */
const EXACT_PROJECTION = 300;

/**
 * Orthogonal projection of b onto the span of `basis` (vectors of b's length). Which vectors are independent is decided
 * exactly. Up to 300 numbers (dimension × basis size) p, e and the coordinates are exact for the doubles given and
 * rounded once; beyond that, modified Gram–Schmidt QR with re-orthogonalisation (backward stable: error about 1e-16
 * times the basis's condition number), falling back to exact arithmetic if QR loses an exactly independent vector.
 */
export function project(b: number[], basis: number[][]): Projection {
  // which basis vectors are exactly independent (in order): columns of the matrix whose rows are the coordinates
  const dim = b.length;
  const rows = Array.from({ length: dim }, (_, i) => basis.map((v) => v[i]));
  const keep = new Set(rref(rows).pivots);
  if (dim * basis.length <= EXACT_PROJECTION) return projectExact(b, basis, [...keep].sort((i, j) => i - j));
  const { Q, R, rank } = qr(basis, keep);
  if (rank < keep.size) return projectExact(b, basis, [...keep].sort((i, j) => i - j));
  const qtb = Q.map((q) => q.reduce((s, x, i) => s + x * b[i], 0));
  const p = b.map((_, i) => Q.reduce((s, q, k) => s + qtb[k] * q[i], 0));
  const e = b.map((x, i) => x - p[i]);
  const n = basis.length;
  const coords = new Array(n).fill(NaN);
  const independent = rank === n;
  if (independent) {
    for (let k = n - 1; k >= 0; k--) {
      let s = qtb[k];
      for (let j = k + 1; j < n; j++) s -= R[k][j] * coords[j];
      coords[k] = s / R[k][k];
    }
  }
  return { p, e, coords, independent };
}

/**
 * The projection in exact arithmetic, for when floating-point QR loses a column that is exactly independent (columns
 * (3, 4) and (0.6, 0.8): their determinant as doubles is 2⁻⁵², so they span the plane, but Gram–Schmidt rounds the
 * second to zero). Solves the normal equations over Q for the independent columns `keep`; every output rounded once.
 */
function projectExact(b: number[], basis: number[][], keep: number[]): Projection {
  // each vector over one power-of-two denominator, so dot products are integer sums with a single reduction at the end
  const scaled = (v: number[]) => {
    const qs = v.map(ex);
    const D = qs.reduce((m, q) => (q.d > m ? q.d : m), 1n); // denominators of doubles are powers of two
    return { ints: qs.map((q) => q.n * (D / q.d)), D };
  };
  const cols = keep.map((j) => scaled(basis[j])), B = scaled(b);
  const dot = (u: { ints: bigint[]; D: bigint }, v: { ints: bigint[]; D: bigint }) => {
    let s = 0n;
    for (let i = 0; i < u.ints.length; i++) s += u.ints[i] * v.ints[i];
    return Q.of(s, u.D * v.D);
  };
  const { R } = rrefQ(cols.map((ci) => [...cols.map((cj) => dot(ci, cj)), dot(ci, B)]));
  const c = R.map((row) => row[cols.length]);
  // p = Σ cₖ colₖ over the common denominator L·D(colₖ) of the coefficients
  const L = c.reduce((m, q) => (m * q.d) / gcdBig(m, q.d), 1n);
  const coef = c.map((q, k) => q.n * (L / q.d) * (cols.reduce((m, col) => (col.D > m ? col.D : m), 1n) / cols[k].D));
  const Dc = cols.reduce((m, col) => (col.D > m ? col.D : m), 1n);
  const P = B.ints.map((_, i) => {
    let s = 0n;
    for (let k = 0; k < cols.length; k++) s += coef[k] * cols[k].ints[i];
    return Q.of(s, L * Dc);
  });
  const coords = new Array(basis.length).fill(NaN);
  const independent = keep.length === basis.length;
  if (independent) keep.forEach((j, k) => (coords[j] = c[k].toNumber()));
  return { p: P.map((x) => x.toNumber()), e: B.ints.map((x, i) => Q.of(x, B.D).sub(P[i]).toNumber()), coords, independent };
}
const gcdBig = (a: bigint, b: bigint): bigint => (b === 0n ? (a < 0n ? -a : a) : gcdBig(b, a % b));

export interface LineFit {
  /** intercept c and slope m of y = c + m x; NaN when the x values are all equal */
  c: number;
  m: number;
  /** fitted values (unique even when c, m are not) and residuals y − fitted */
  fitted: number[];
  residuals: number[];
  sse: number;
  /** false when every x is equal: the fit is the mean line through x̄ and c + x̄m = ȳ has a line of solutions */
  unique: boolean;
}

/**
 * Least-squares line by the centred formulas (no normal equations, so large x offsets do not square the condition).
 * Values are first taken relative to the first point (exact when the points are within a factor of 2 of it, the
 * large-offset case), so the mean is never rounded at the offset's scale: x = 1e16 and 1e16 + 2 have the mean
 * 1e16 + 1, which is not a double.
 */
export function fitLine(xs: number[], ys: number[]): LineFit {
  const n = xs.length;
  const x0 = xs[0], y0 = ys[0];
  const dx = xs.map((x) => x - x0), dy = ys.map((y) => y - y0);
  const mx = dx.reduce((s, x) => s + x, 0) / n, my = dy.reduce((s, y) => s + y, 0) / n;
  if (xs.every((x) => x === x0)) {
    // decided on the values given, not on a rounded mean (three copies of 0.1 average to 0.10000000000000002)
    const residuals = dy.map((v) => v - my);
    return { c: NaN, m: NaN, fitted: residuals.map(() => y0 + my), residuals, sse: residuals.reduce((s, r) => s + r * r, 0), unique: false };
  }
  // the centred values scaled by powers of two (exact) before squaring, so tiny or huge data neither under- nor overflow
  const us = dx.map((u) => u - mx), vs = dy.map((v) => v - my);
  const p2 = (a: number[]) => {
    const t = Math.max(...a.map(Math.abs));
    return t > 0 && Number.isFinite(t) ? Math.round(Math.log2(t)) : 0;
  };
  const kx = p2(us), ky = p2(vs);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const u = scale2(us[i], -kx), v = scale2(vs[i], -ky);
    sxx += u * u;
    sxy += u * v;
  }
  const m = scale2(sxy / sxx, ky - kx);
  const rel = dx.map((u) => my + m * (u - mx)); // fitted, relative to y0
  const residuals = dy.map((v, i) => v - rel[i]);
  // c = ȳ − m·x̄ = y0 + my − m·x0 − m·mx, formed exactly from those doubles and rounded once (m·x0 alone can round by
  // more than c is worth when the offset is large: x = 1e16, 1e16 + 2 and y = 1e16, 1e16 + 4 have c = −1e16 exactly)
  const M = Number.isFinite(m) ? ex(m) : null;
  const c = M ? ex(y0).add(ex(my)).sub(M.mul(ex(x0))).sub(M.mul(ex(mx))).toNumber() : NaN;
  return { c, m, fitted: rel.map((v) => y0 + v), residuals, sse: residuals.reduce((s, r) => s + r * r, 0), unique: true };
}

/** Least squares for A x ≈ b with A given by its columns (any number of rows), through QR. */
export function leastSquares(columns: number[][], b: number[]): Projection {
  return project(b, columns);
}

// ---------------------------------------------------------------------------------------------------------------
// 4 × 4 rotations (the tesseract page): a rotation in 4D turns a plane, not about an axis. Row-major like M2 / M3.

export type V4 = [number, number, number, number];
/** A 4 × 4 matrix, row-major (16 entries). */
export type M4 = number[];

export const id4 = (): M4 => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** The rotation by θ (radians) in the plane of axes a and b (0 = x … 3 = w): a turns toward b, every axis outside
 *  the plane stays put. cos and sin are taken exactly at the quarter turns (θ = kπ/2), so a quarter turn is an exact
 *  permutation with signs. */
export function planeRot4(a: number, b: number, theta: number): M4 {
  const q = theta / (Math.PI / 2);
  const k = Math.round(q);
  const exact = Math.abs(q - k) < 1e-12;
  const c = exact ? [1, 0, -1, 0][((k % 4) + 4) % 4] : Math.cos(theta);
  const s = exact ? [0, 1, 0, -1][((k % 4) + 4) % 4] : Math.sin(theta);
  const M = id4();
  M[4 * a + a] = c; M[4 * a + b] = -s;
  M[4 * b + a] = s; M[4 * b + b] = c;
  return M;
}
export function mul4(A: M4, B: M4): M4 {
  const C = new Array<number>(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += A[4 * i + k] * B[4 * k + j];
    C[4 * i + j] = s;
  }
  return C;
}
export const apply4 = (M: M4, v: readonly number[]): V4 => [
  M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[3] * v[3],
  M[4] * v[0] + M[5] * v[1] + M[6] * v[2] + M[7] * v[3],
  M[8] * v[0] + M[9] * v[1] + M[10] * v[2] + M[11] * v[3],
  M[12] * v[0] + M[13] * v[1] + M[14] * v[2] + M[15] * v[3],
];
