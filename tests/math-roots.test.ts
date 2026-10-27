import { describe, expect, it } from 'vitest';
import { bisect, brent, goldenMax, lobeCrossings, scanRoots } from '../src/lib/math/roots';

describe('bracketed roots', () => {
  it('bisection and Brent find √2 and the Dottie number to the last bit', () => {
    for (const solve of [bisect, brent]) {
      const r = solve((x) => x * x - 2, 0, 2);
      expect(r.ok).toBe(true);
      expect(Math.abs(r.x - Math.SQRT2)).toBeLessThanOrEqual(4.5e-16);
      const d = solve((x) => Math.cos(x) - x, 0, 1);
      expect(Math.abs(d.x - 0.7390851332151607)).toBeLessThanOrEqual(2.3e-16);
    }
  });
  it('Brent needs far fewer evaluations than bisection on a smooth function', () => {
    const f = (x: number) => Math.exp(x) - 3;
    expect(brent(f, 0, 2).iterations).toBeLessThan(bisect(f, 0, 2).iterations / 3);
  });
  it('without a sign change: ok false and NaN, never a guess', () => {
    for (const solve of [bisect, brent]) {
      const r = solve((x) => x * x + 1, -1, 2);
      expect(r.ok).toBe(false);
      expect(r.x).toBeNaN();
    }
  });
  it('a root at an end of the bracket is returned as is', () => {
    expect(brent((x) => x - 1, 1, 3)).toMatchObject({ x: 1, ok: true });
    expect(bisect((x) => x - 3, 1, 3)).toMatchObject({ x: 3, ok: true });
  });
  it('a sign change without a root (a pole) is bracketed, not confused with a root: the caller checks |f|', () => {
    const r = bisect((x) => 1 / (x - 0.3), 0, 1);
    expect(Math.abs(r.x - 0.3)).toBeLessThan(1e-15);
    expect(Math.abs(1 / (r.x - 0.3))).toBeGreaterThan(1e14);
  });
});

describe('scanning and maxima', () => {
  it('scanRoots finds well-separated roots', () => {
    const rs = scanRoots(Math.sin, 0.5, 10, 100);
    expect(rs.length).toBe(3);
    rs.forEach((x, k) => expect(Math.abs(x - (k + 1) * Math.PI)).toBeLessThan(1e-14));
  });
  it('scanRoots can miss two roots inside one step (documented): use a bracket or lobeCrossings there', () => {
    expect(scanRoots((x) => (x - 1.05) * (x - 1.051), 0, 2, 10)).toEqual([]); // both between the samples 1.0 and 1.2
  });
  it('goldenMax on a unimodal function', () => {
    const m = goldenMax(Math.sin, 0, Math.PI);
    expect(Math.abs(m.x - Math.PI / 2)).toBeLessThan(1e-7); // flat top: x to √ε, the value to ε
    expect(Math.abs(m.value - 1)).toBeLessThan(1e-15);
  });
});

describe('lobeCrossings: the D01 largest δ for x·sin(1/x)', () => {
  // largest δ with |x sin(1/x)| < ε for 0 < x < δ: the first x (from 0 outward) where |x sin(1/x)| reaches ε.
  // Nothing fails below x = ε (|x sin(1/x)| ≤ |x|), so the lobes [1/((k+1)π), 1/(kπ)] are examined from just below ε.
  function largestDelta(eps: number) {
    const K = Math.floor(1 / (eps * Math.PI)) + 2;
    const edges: number[] = [];
    for (let k = K; k >= 1; k--) edges.push(1 / (k * Math.PI));
    const g = (x: number) => Math.abs(x * Math.sin(1 / x)) - eps;
    const [first] = lobeCrossings(g, edges, { firstOnly: true });
    return first.rising;
  }
  it('ε = 0.1 → δ = 0.1187196710; ε = 1e-4 → δ = 0.0001000186243 (D01 v0.2 values)', () => {
    expect(Math.abs(largestDelta(0.1) - 0.118719671)).toBeLessThan(1e-9);
    expect(Math.abs(largestDelta(1e-4) - 0.0001000186243)).toBeLessThan(1e-12);
  });
  it('the δ is never below ε, and points just inside it are within ε', () => {
    for (const eps of [0.1, 1e-2, 1e-3]) {
      const d = largestDelta(eps);
      expect(d).toBeGreaterThanOrEqual(eps);
      for (let i = 1; i <= 2000; i++) {
        const x = (d * i) / 2001;
        expect(Math.abs(x * Math.sin(1 / x))).toBeLessThan(eps);
      }
    }
  });
  it('a lobe that never reaches 0 is skipped, every crossing lobe reported without firstOnly', () => {
    const edges = [1 / (5 * Math.PI), 1 / (4 * Math.PI), 1 / (3 * Math.PI), 1 / (2 * Math.PI), 1 / Math.PI];
    const all = lobeCrossings((x) => Math.abs(x * Math.sin(1 / x)) - 0.1, edges);
    expect(all.map((c) => c.lobe)).toEqual([2, 3]);
    for (const c of all) expect(c.rising).toBeLessThan(c.falling);
  });
});

describe('Codex review 2026-10-07: regressions', () => {
  it('#19 a bracket spanning the whole double range, and NaN inside the bracket', () => {
    expect(bisect((x) => x, -1e308, 1e308)).toMatchObject({ x: 0, ok: true });
    expect(brent((x) => x - 1, -1e308, 1e308)).toMatchObject({ ok: true });
    expect(Math.abs(brent((x) => x - 1, -1e308, 1e308).x - 1)).toBeLessThan(1e-15);
    const r = bisect((x) => (x > 0.3 && x < 0.32 ? NaN : x - 0.35), 0, 1); // the fourth midpoint, 0.3125, is NaN
    expect(r.ok).toBe(false);
  });
});

describe('Codex re-review 2026-10-07: regressions of the first fixes', () => {
  it('#19 Brent stops when the bracket is two adjacent (subnormal) doubles', () => {
    const u = Number.MIN_VALUE;
    const r = brent((x) => x / u - 2.5, 2 * u, 3 * u);
    expect(r.ok).toBe(true);
    expect([2 * u, 3 * u]).toContain(r.x);
    expect(r.iterations).toBeLessThan(5);
  });
});
