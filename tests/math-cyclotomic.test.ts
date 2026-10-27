// src/lib/math/cyclotomic.ts: exact zero tests for rational sums of cosines and sines of rational angles.
import { describe, expect, it } from 'vitest';
import { Q } from '../src/lib/math/rational';
import { MAX_RADICAL, rootOfUnitySumIsZero, trigIsZero, trigSumIsZero, type TrigTerm } from '../src/lib/math/cyclotomic';

const q = (n: number, d = 1) => Q.of(n, d);
const cos = (deg: Q, a: Q = Q.ONE): TrigTerm => ({ deg, cos: a, sin: Q.ZERO });
const sin = (deg: Q, b: Q = Q.ONE): TrigTerm => ({ deg, cos: Q.ZERO, sin: b });

describe('trigSumIsZero: identities hold exactly', () => {
  it('values with a rational cosine or sine', () => {
    expect(trigIsZero(q(-1, 2), q(1), q(0), q(60))).toBe(true); // cos 60° = ½
    expect(trigIsZero(q(-1, 2), q(0), q(1), q(30))).toBe(true); // sin 30° = ½
    expect(trigIsZero(q(0), q(1), q(0), q(90))).toBe(true); // cos 90° = 0
    expect(trigIsZero(q(1), q(1), q(0), q(180))).toBe(true); // cos 180° = −1
    expect(trigIsZero(q(0), q(1), q(-1), q(45))).toBe(true); // cos 45° = sin 45°
    expect(trigIsZero(q(1, 2), q(1), q(0), q(-240))).toBe(true); // cos(−240°) = −½
  });
  it('the golden ratio: cos 36° − cos 72° = ½', () => {
    expect(trigSumIsZero(q(-1, 2), [cos(q(36)), cos(q(72), q(-1))])).toBe(true);
  });
  it('sevenths: cos(π/7) − cos(2π/7) + cos(3π/7) = ½', () => {
    expect(trigSumIsZero(q(-1, 2), [cos(q(180, 7)), cos(q(360, 7), q(-1)), cos(q(540, 7))])).toBe(true);
  });
  it('sin 18° = (√5 − 1)/4, so 4 sin 18° + 1 − 4 cos 36° + 1 … : 2 sin 18° = 2 cos 36° − 1', () => {
    expect(trigSumIsZero(q(1), [sin(q(18), q(2)), cos(q(36), q(-2))])).toBe(true);
  });
  it('a sum over a full set of directions is 0: Σ cos(k·72°) = 0', () => {
    expect(trigSumIsZero(q(0), [0, 72, 144, 216, 288].map((d) => cos(q(d))))).toBe(true);
  });
  it('terms at the same angle are added before the test (12.345° ± 0.001°)', () => {
    const a = q(12345, 1000), b = q(1, 1000);
    expect(trigSumIsZero(q(0), [sin(a.add(b)), sin(a.add(b), q(-1, 2)), sin(a.sub(b), q(-1, 2)), sin(a.add(b), q(-1, 2)), sin(a.sub(b), q(1, 2))])).toBe(true);
  });
  it('a direction stated twice cancels: cos β − sin(90° − β)', () => {
    expect(trigSumIsZero(q(0), [cos(q(1234567, 1000)), sin(q(90).sub(q(1234567, 1000)), q(-1))])).toBe(true);
  });
});

describe('trigSumIsZero: near misses are not zero', () => {
  it('89.9999°, 90.001°', () => {
    expect(trigIsZero(q(0), q(1), q(0), q(899999, 10000))).toBe(false);
    expect(trigIsZero(q(0), q(1), q(0), q(90001, 1000))).toBe(false);
  });
  it('cos 1° is not rational, cos 15° is not √3/2 − …', () => {
    expect(trigIsZero(q(-1), q(1), q(0), q(1))).toBe(false);
    expect(trigIsZero(q(0), q(1), q(-1), q(15))).toBe(false);
    expect(trigIsZero(q(-1, 2), q(1), q(0), q(60001, 1000))).toBe(false);
  });
  it('a tiny rational offset breaks an identity', () => {
    expect(trigSumIsZero(q(-1, 2).add(q(1, 10 ** 15)), [cos(q(36)), cos(q(72), q(-1))])).toBe(false);
  });
});

describe('trigSumIsZero agrees with floating point where the value is clearly nonzero, and finds the zeros', () => {
  // a seeded generator, so the test is the same every run
  let s = 12345;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  it('400 random sums in thousandths of a degree', () => {
    for (let k = 0; k < 400; k++) {
      const terms: TrigTerm[] = [];
      let value = 0;
      const n = 1 + Math.floor(rnd() * 3);
      for (let i = 0; i < n; i++) {
        const deg = q(Math.floor(rnd() * 720000) - 360000, 1000);
        const a = q(Math.floor(rnd() * 21) - 10, 1 + Math.floor(rnd() * 4));
        const b = q(Math.floor(rnd() * 21) - 10, 1 + Math.floor(rnd() * 4));
        terms.push({ deg, cos: a, sin: b });
        const r = (deg.toNumber() * Math.PI) / 180;
        value += a.toNumber() * Math.cos(r) + b.toNumber() * Math.sin(r);
      }
      // c chosen to cancel exactly when the angles are multiples of 90° (all cos, sin in {0, ±1}), else arbitrary
      const nice = terms.every((t) => t.deg.mul(q(1, 90)).isInteger());
      // cos and sin of a multiple of 90°, exactly
      const quarter = (t: TrigTerm) => Number(((t.deg.mul(q(1, 90)).n % 4n) + 4n) % 4n);
      const exactSum = () => terms.reduce((acc, t) => acc.add(t.cos.mul(q([1, 0, -1, 0][quarter(t)]))).add(t.sin.mul(q([0, 1, 0, -1][quarter(t)]))), Q.ZERO);
      const c = nice ? exactSum().neg() : q(Math.floor(rnd() * 11) - 5, 3);
      const exact = trigSumIsZero(c, terms);
      if (nice) expect(exact).toBe(true);
      else if (Math.abs(value + c.toNumber()) > 1e-9) expect(exact).toBe(false);
    }
  });
});

describe('rootOfUnitySumIsZero: its own entry', () => {
  it('exponents are taken mod N: ζ₄⁻¹ + ζ₄ = −i + i = 0, and ζ₄⁵ is ζ₄', () => {
    expect(rootOfUnitySumIsZero(4n, new Map([[-1n, 1n], [1n, 1n]]))).toBe(true);
    expect(rootOfUnitySumIsZero(4n, new Map([[5n, 1n], [3n, 1n]]))).toBe(true);
    expect(rootOfUnitySumIsZero(4n, new Map([[5n, 1n], [1n, 1n]]))).toBe(false);
  });
  it('a prime factor beyond MAX_RADICAL is refused at once, not searched for', () => {
    const big = 1000003n; // prime
    expect(big > BigInt(MAX_RADICAL)).toBe(true);
    const t0 = performance.now();
    expect(() => trigIsZero(Q.ZERO, Q.ONE, Q.ZERO, Q.of(360n, big))).toThrow(RangeError);
    expect(performance.now() - t0).toBeLessThan(50);
  });
});
