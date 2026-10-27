import { describe, expect, it } from 'vitest';
import { approximate, exactDecimal, isqrt, niceStepQ, parseDecimal, parseExact, parseQText, Q, roundTo, shownDecimal, splitSquare, texOf } from '../src/lib/math/rational';

describe('Q: exact rationals', () => {
  it('normalises sign and lowest terms; arithmetic is exact', () => {
    const a = Q.of(6, -4);
    expect([a.n, a.d]).toEqual([-3n, 2n]);
    const third = Q.of(1, 3), sixth = Q.of(1, 6);
    expect(third.add(sixth).eq(Q.of(1, 2))).toBe(true);
    expect(third.sub(sixth).mul(Q.of(6)).eq(Q.ONE)).toBe(true);
    expect(Q.of(2, 3).div(Q.of(4, 9)).eq(Q.of(3, 2))).toBe(true);
    expect(() => Q.ONE.div(Q.ZERO)).toThrow(RangeError);
    expect(Q.of(-7, 2).cmp(Q.of(-3))).toBe(-1);
    expect(Q.of(2, 3).pow(-2).eq(Q.of(9, 4))).toBe(true);
    expect(Q.of(-5, 3).inv().eq(Q.of(-3, 5))).toBe(true);
    expect([Q.of(7, 2).floor(), Q.of(-7, 2).floor(), Q.of(4).floor()]).toEqual([3n, -4n, 4n]);
    expect(Q.min(Q.of(1, 3), Q.of(1, 4)).eq(Q.of(1, 4))).toBe(true);
    expect(Q.of(9, 4).sqrtExact()?.eq(Q.of(3, 2))).toBe(true);
    expect(Q.of(2).sqrtExact()).toBeNull();
  });
  it('fromDouble is the exact binary value: 0.1 is 3602879701896397 / 2^55', () => {
    const q = Q.fromDouble(0.1);
    expect(q.n).toBe(3602879701896397n);
    expect(q.d).toBe(2n ** 55n);
    expect(q.toNumber()).toBe(0.1);
    // 0.1 + 0.2 ≠ 0.3 for doubles, and Q sees why: the exact sum of the two doubles is not the double 0.3
    expect(Q.fromDouble(0.1).add(Q.fromDouble(0.2)).eq(Q.fromDouble(0.3))).toBe(false);
    for (const x of [0, -2.5, 1e-300, 123456.789, 2 ** 60, -Number.MIN_VALUE]) expect(Q.fromDouble(x).toNumber()).toBe(x);
  });
  it('toNumber is correctly rounded at any size', () => {
    expect(Q.of(10n ** 400n + 1n, 10n ** 399n).toNumber()).toBe(10);
    expect(Q.of(1n, 10n ** 400n).toNumber()).toBe(0);
    expect(Q.of(10n ** 400n).toNumber()).toBe(Infinity);
    // every double survives the round trip, normal and subnormal (a deterministic sweep of exponents and mantissas)
    let seed = 12345;
    const next = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
    for (let i = 0; i < 3000; i++) {
      const x = (next() - 0.5) * 2 ** Math.floor(next() * 2098 - 1074);
      if (x === 0) continue; // an underflow to −0: a rational has no sign of zero
      expect(Q.fromDouble(x).toNumber()).toBe(x);
    }
    // p/q with exact p, q is correctly rounded by the division operator, so the two must agree
    for (let i = 0; i < 3000; i++) {
      const p = Math.floor(next() * 2 ** 40) - 2 ** 39, q = Math.floor(next() * 2 ** 40) + 1;
      expect(Q.of(p, q).toNumber()).toBe(p / q);
      expect(Q.of(BigInt(p) * 10n ** 300n, BigInt(q) * 10n ** 300n).toNumber()).toBe(p / q);
    }
  });
});

describe('helpers', () => {
  it('approximate: a fraction label for a computed double, exact only when it is', () => {
    expect(approximate(0.5)).toEqual({ q: Q.of(1, 2), exact: true });
    const t = approximate(1 / 3)!;
    expect(t.q.eq(Q.of(1, 3))).toBe(true);
    expect(t.exact).toBe(true); // the double 1/3 is what 1/3 rounds to
    const p = approximate(Math.PI, 1000)!;
    expect(p.q.eq(Q.of(355, 113))).toBe(true);
    expect(p.exact).toBe(false);
    expect(approximate(Math.PI, 100)!.q.eq(Q.of(311, 99))).toBe(true); // best with denominator ≤ 100
    expect(approximate(-0.75)!.q.eq(Q.of(-3, 4))).toBe(true);
    expect(approximate(NaN)).toBeNull();
  });
  it('texOf puts the sign outside the fraction', () => {
    expect(texOf(Q.of(-3, 4))).toBe('-\\frac{3}{4}');
    expect(texOf(Q.of(5))).toBe('5');
  });
  it('parseExact, exactDecimal, roundTo, niceStepQ', () => {
    expect(parseExact('0.1')!.eq(Q.of(1, 10))).toBe(true);
    expect(parseExact(' −3/4 ')!.eq(Q.of(-3, 4))).toBe(true);
    expect(parseExact('1/0')).toBeNull();
    expect(parseExact('abc')).toBeNull();
    expect(exactDecimal(Q.of(5, 4))).toBe('1.25');
    expect(exactDecimal(Q.of(1, 3))).toBeNull();
    expect(roundTo(0.29999, Q.of(1, 10)).eq(Q.of(3, 10))).toBe(true);
    for (const raw of [0.003, 0.07, 0.4, 3, 70]) {
      const s = niceStepQ(raw).toNumber();
      expect(s).toBeGreaterThanOrEqual(0.01);
      expect([1, 2, 5]).toContain(Math.round(s / 10 ** Math.floor(Math.log10(s) + 1e-9)));
    }
  });
  it('integer square roots and square splitting', () => {
    expect(isqrt(10n ** 30n)).toBe(10n ** 15n);
    expect(isqrt(99n)).toBe(9n);
    expect(splitSquare(72n)).toEqual({ s: 6n, t: 2n }); // √72 = 6√2
  });
});

describe('shownDecimal and parseQText (F08, 2026-10-07)', () => {
  it('a double means the decimal it prints as, up to six places', () => {
    expect(shownDecimal(0.1)!.eq(Q.of(1, 10))).toBe(true);
    expect(shownDecimal(-2.75)!.eq(Q.of(-11, 4))).toBe(true);
    expect(shownDecimal(0.000001)!.eq(Q.of(1, 1000000))).toBe(true);
    expect(shownDecimal(1 / 3)).toBeNull();
    expect(shownDecimal(1e-7)).toBeNull(); // prints as 1e-7
    expect(shownDecimal(Infinity)).toBeNull();
  });
  it('Q.toString read back exactly at any size', () => {
    const big = Q.of(1n, 10n ** 20n);
    expect(parseQText(big.toString())!.eq(big)).toBe(true);
    expect(parseQText('-5/4')!.eq(Q.of(-5, 4))).toBe(true);
    expect(parseQText('−3')!.eq(Q.of(-3))).toBe(true);
    expect(parseQText('1/0')).toBeNull();
    expect(parseQText('0.5')).toBeNull();
  });
});

describe('parseDecimal (F08, 2026-10-07)', () => {
  it('any decimal, exactly, with an exponent', () => {
    expect(parseDecimal('1e-13')!.eq(Q.of(1n, 10n ** 13n))).toBe(true);
    expect(parseDecimal('0.0000001')!.eq(Q.of(1n, 10n ** 7n))).toBe(true);
    expect(parseDecimal('−2.5E3')!.eq(Q.of(-2500))).toBe(true);
    expect(parseDecimal('.5')!.eq(Q.of(1, 2))).toBe(true);
    expect(parseDecimal('1/3')).toBeNull();
    expect(parseDecimal('π')).toBeNull();
    expect(parseDecimal('1e999')).toBeNull();
    expect(parseDecimal('')).toBeNull();
  });
});
