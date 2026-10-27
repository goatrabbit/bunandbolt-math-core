// Surd: exact a + b√t (src/lib/math/surd.ts; D22's slice areas).
import { describe, expect, test } from 'vitest';
import { Q } from '../src/lib/math/rational';
import { Surd, cosOfTurn, sAdd, sMul, sSub, surdForm } from '../src/lib/math/surd';

describe('Surd: a + b√t', () => {
  test('square roots of rationals, exact where they are', () => {
    expect(Surd.sqrt(Q.of(9, 4))!.rational()!.eq(Q.of(3, 2))).toBe(true);
    const r2 = Surd.sqrt(Q.of(2))!;
    expect(r2.isRational()).toBe(false);
    expect(r2.sq().rational()!.eq(Q.of(2))).toBe(true);
    expect(Surd.sqrt(Q.of(8))!.toNumber()).toBeCloseTo(Math.sqrt(8), 14);
    expect(Surd.sqrt(Q.of(-1))).toBeNull();
    // (2 + √3)² − (2 − √3)² = 8√3: irrational; (√2)(√8) = 4
    const a = Surd.of(2).add(Surd.sqrt(Q.of(3))!)!, b = Surd.of(2).sub(Surd.sqrt(Q.of(3))!)!;
    expect(a.sq().sub(b.sq())!.isRational()).toBe(false);
    expect(r2.mul(Surd.sqrt(Q.of(8))!)!.rational()!.eq(Q.of(4))).toBe(true);
    // two different roots cannot be combined: null, never a guess
    expect(r2.add(Surd.sqrt(Q.of(3))!)).toBeNull();
  });
});

describe('Surd: missing values and roots', () => {
  test('helpers pass null on', () => {
    expect(sMul(null, Surd.of(2))).toBeNull();
    expect(sAdd(Surd.of(1), null)).toBeNull();
    expect(sSub(Surd.of(3), Surd.of(1))!.rational()!.eq(Q.of(2))).toBe(true);
  });
  test('√(n/d) = (s/d)√t, and √(9/4) is rational', () => {
    const r = Surd.sqrt(Q.of(1, 2))!; // √2/2
    expect(r.root).toBe(2n);
    expect(r.toNumber()).toBeCloseTo(Math.SQRT1_2, 15);
    expect(r.sq().rational()!.eq(Q.of(1, 2))).toBe(true);
    expect(Surd.sqrt(Q.ZERO)!.rational()!.isZero()).toBe(true);
    // a root part that cancels leaves a rational: (1 + √2) − √2 = 1
    expect(Surd.of(1).add(r.scale(Q.of(2)))!.sub(r.scale(Q.of(2)))!.rational()!.eq(Q.ONE)).toBe(true);
  });
});

describe('Surd: sign, division, exact cosines, short forms (B04)', () => {
  const r = (n: number) => Surd.sqrt(Q.of(n))!;
  test('the exact sign: 9 − 6√3 < 0 < 16 − 6√3, 2 − √4 = 0', () => {
    expect(Surd.of(9).sub(r(3).scale(Q.of(6)))!.sign()).toBe(-1);
    expect(Surd.of(16).sub(r(3).scale(Q.of(6)))!.sign()).toBe(1);
    expect(Surd.of(2).sub(r(4))!.sign()).toBe(0);
    expect(r(2).neg().sign()).toBe(-1);
    expect(Surd.of(-1).add(r(2))!.sign()).toBe(1);
  });
  test('division by the conjugate', () => {
    // (25 − 12√3)/(2 − √3) = (25 − 12√3)(2 + √3) = 14 + √3
    const q = Surd.of(25).sub(r(3).scale(Q.of(12)))!.div(Surd.of(2).sub(r(3))!)!;
    expect(q.eq(Surd.of(14).add(r(3))!)).toBe(true);
    expect(Surd.of(6).div(r(2))!.eq(r(2).scale(Q.of(3)))).toBe(true);
    expect(Surd.of(1).div(Surd.of(0))).toBeNull();
    expect(r(2).div(r(3))).toBeNull(); // two roots
  });
  test('cos at the turns where it is a + b√t; null elsewhere', () => {
    const deg = (d: number) => cosOfTurn(Q.of(d, 360));
    for (const d of [0, 30, 36, 45, 60, 72, 90, 108, 120, 135, 144, 150, 180, -60, 300, 390]) {
      const c = deg(d);
      expect(c).not.toBeNull();
      expect(c!.toNumber()).toBeCloseTo(Math.cos((d * Math.PI) / 180), 14);
    }
    expect(deg(90)!.isZero()).toBe(true);
    expect(deg(60)!.rational()!.eq(Q.of(1, 2))).toBe(true);
    for (const d of [15, 18, 22.5, 75, 1, 89, 91]) expect(cosOfTurn(Q.fromDouble(d).div(Q.of(360)))).toBeNull();
  });
  test('short forms for readouts', () => {
    expect(surdForm(Surd.of(9).sub(r(3).scale(Q.of(6)))!)).toBe('9 − 6√3');
    expect(surdForm(r(13))).toBe('√13');
    expect(surdForm(r(3).scale(Q.of(1, 2)))).toBe('√3/2');
    expect(surdForm(r(3).scale(Q.of(1, 2)), 4, true)).toBe('\\tfrac{\\sqrt{3}}{2}');
    expect(surdForm(Surd.of(Q.of(729, 100)))).toBe('7.29');
    expect(surdForm(Surd.of(Q.of(-1, 3)))).toBe('−1/3');
    expect(surdForm(Surd.of(-9).add(r(3).scale(Q.of(25, 4)))!)).toBe('25√3/4 − 9');
    expect(surdForm(Surd.of(-9).sub(r(3))!)).toBe('−9 − √3');
    expect(surdForm(Surd.of(Q.of(1, 7919)))).toBeNull();
    expect(surdForm(Surd.of(Q.of(12345, 1000)), 2)).toBeNull();
  });
});
