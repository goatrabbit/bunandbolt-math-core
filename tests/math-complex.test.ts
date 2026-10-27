// src/lib/math/complex.ts: the exact parts a complex number carries survive exactly as far as the mathematics allows
// (2026-10-07, A15 complex plane).
import { describe, expect, it } from 'vitest';
import { Q } from '../src/lib/math/rational';
import {
  argDeg, cxAdd, cxMul, cxNum, cxPolar, cxPow, cxRect, exactParts, I, isZero, modulus, ONE, principalTurn, provablyEqual, realQ, rectOfPolar, surd, turnOfRect, turnQ, ZERO,
} from '../src/lib/math/complex';

const q = (n: number, d = 1) => Q.of(n, d);
const txt = (x: Q | null) => (x ? x.toString() : null);
const rectTxt = (c: { rect: readonly [Q, Q] | null }) => (c.rect ? `${c.rect[0]} ${c.rect[1]}` : null);

describe('principal turns', () => {
  it('reduce into (−1/2, 1/2]', () => {
    expect(txt(principalTurn(q(1, 2)))).toBe('1/2');
    expect(txt(principalTurn(q(-1, 2)))).toBe('1/2');
    expect(txt(principalTurn(q(11, 18)))).toBe('-7/18'); // 220° → −140°
    expect(txt(principalTurn(q(1)))).toBe('0');
    expect(txt(principalTurn(q(-5, 4)))).toBe('-1/4');
  });
});

describe('the turn of a rational point is exact only on the axes and the diagonals', () => {
  it('axes and diagonals', () => {
    expect(txt(turnOfRect(q(2), q(0)))).toBe('0');
    expect(txt(turnOfRect(q(-3), q(0)))).toBe('1/2');
    expect(txt(turnOfRect(q(0), q(1, 2)))).toBe('1/4');
    expect(txt(turnOfRect(q(0), q(-1)))).toBe('-1/4');
    expect(txt(turnOfRect(q(1), q(1)))).toBe('1/8');
    expect(txt(turnOfRect(q(-2), q(2)))).toBe('3/8');
    expect(txt(turnOfRect(q(-1), q(-1)))).toBe('-3/8');
  });
  it('elsewhere none (2 + i has no exact number of degrees)', () => {
    expect(turnOfRect(q(2), q(1))).toBeNull();
    expect(turnOfRect(q(0), q(0))).toBeNull();
  });
});

describe('rational parts from |c|² and a special turn', () => {
  it('finds them where they are rational', () => {
    expect(rectOfPolar(q(2), q(1, 8))!.map(String)).toEqual(['1', '1']); // √2∠45° = 1 + i
    expect(rectOfPolar(q(4), q(1, 6))).toBeNull(); // 2∠60° = 1 + √3 i
    expect(rectOfPolar(q(4), q(1, 3))).toBeNull();
    expect(rectOfPolar(q(16, 3), q(1, 12))).toBeNull(); // re² = 4, im² = 4/3
    expect(rectOfPolar(q(4), q(-1, 3))).toBeNull(); // 2∠−120° = −1 − √3 i
    expect(rectOfPolar(q(1), q(1, 4))!.map(String)).toEqual(['0', '1']);
    expect(rectOfPolar(q(9), q(1, 2))!.map(String)).toEqual(['-3', '0']);
    expect(rectOfPolar(q(8), q(-3, 8))!.map(String)).toEqual(['-2', '-2']);
  });
  it('none at other turns or irrational parts', () => {
    expect(rectOfPolar(q(1), q(1, 5))).toBeNull(); // 72°
    expect(rectOfPolar(q(1), q(1, 6))).toBeNull(); // 60°: 1/2 + (√3/2)i
    expect(rectOfPolar(q(1), q(1, 24))).toBeNull(); // 15°
  });
});

describe('cxRect, cxPolar', () => {
  it('rect knows |c|² and, on axes and diagonals, the turn', () => {
    const z = cxRect(q(2), q(1));
    expect(txt(z.mod2)).toBe('5');
    expect(z.turn).toBeNull();
    expect(txt(cxRect(q(1), q(1)).turn)).toBe('1/8');
  });
  it('polar from exact data: value from the exact turn, rect where it exists', () => {
    const w = cxPolar(realQ(q(3, 2)), turnQ(q(1, 9))); // 1.5∠40°
    expect(txt(w.mod2)).toBe('9/4');
    expect(txt(w.turn)).toBe('1/9');
    expect(w.rect).toBeNull();
    expect(w.re).toBeCloseTo(1.5 * Math.cos((40 * Math.PI) / 180), 14);
    const i = cxPolar(realQ(q(1)), turnQ(q(1, 4)));
    expect(rectTxt(i)).toBe('0 1');
    expect(i.re).toBe(0);
  });
  it('a polar turn is reduced to the principal value', () => {
    expect(txt(cxPolar(realQ(q(1)), turnQ(q(3, 4))).turn)).toBe('-1/4');
  });
  it('an exact zero modulus is exactly 0', () => {
    const z = cxPolar(realQ(q(0)), turnQ(q(1, 9)));
    expect(rectTxt(z)).toBe('0 0');
    expect(z.turn).toBeNull();
    expect(isZero(z)).toBe(true);
  });
  it('refuses a negative modulus', () => {
    expect(() => cxPolar({ v: -1, sq: q(1) }, turnQ(q(0)))).toThrow();
  });
});

describe('products', () => {
  it('(2 + i)(1 + i) = 1 + 3i exactly; |zw|² = 10 = 5 · 2', () => {
    const p = cxMul(cxRect(q(2), q(1)), cxRect(q(1), q(1)));
    expect(rectTxt(p)).toBe('1 3');
    expect(txt(p.mod2)).toBe('10');
    expect(p.turn).toBeNull();
  });
  it('i·i = −1, i(2 + i) = −1 + 2i', () => {
    expect(rectTxt(cxMul(I, I))).toBe('-1 0');
    expect(rectTxt(cxMul(I, cxRect(q(2), q(1))))).toBe('-1 2');
  });
  it('turns add (mod a whole turn): 120° + 100° → −140°', () => {
    const p = cxMul(cxPolar(realQ(q(2)), turnQ(q(1, 3))), cxPolar(realQ(q(1)), turnQ(q(5, 18))));
    expect(txt(p.turn)).toBe('-7/18');
    expect(argDeg(p)!.exact!.toString()).toBe('-140');
    expect(txt(p.mod2)).toBe('4');
  });
  it('mixed: rect × polar keeps |·|² and drops the turn', () => {
    const p = cxMul(cxRect(q(2), q(1)), cxPolar(realQ(q(3, 2)), turnQ(q(1, 9))));
    expect(txt(p.mod2)).toBe('45/4'); // |zw| = 3√5/2
    expect(p.turn).toBeNull();
    expect(p.rect).toBeNull();
    expect(modulus(p)).toBeCloseTo((3 * Math.sqrt(5)) / 2, 14);
  });
  it('numeric × exact claims nothing exact', () => {
    const p = cxMul(cxNum(1.2, 0.3), I);
    expect(p.rect).toBeNull();
    expect(p.mod2).toBeNull();
    expect(p.turn).toBeNull();
    expect(p.re).toBeCloseTo(-0.3, 15);
  });
  it('0 times anything is exactly 0', () => {
    expect(rectTxt(cxMul(ZERO, cxNum(3, 4)))).toBe('0 0');
    expect(rectTxt(cxMul(cxPolar(realQ(q(1)), turnQ(q(1, 5))), ZERO))).toBe('0 0');
  });
  it('conjugates: (2 + i)(2 − i) = 5', () => {
    expect(rectTxt(cxMul(cxRect(q(2), q(1)), cxRect(q(2), q(-1))))).toBe('5 0');
  });
});

describe('sums', () => {
  it('rect + rect exactly; |z + w| ≤ |z| + |w|', () => {
    const z = cxRect(q(2), q(1)), w = cxRect(q(1), q(1));
    const s = cxAdd(z, w);
    expect(rectTxt(s)).toBe('3 2');
    expect(txt(s.mod2)).toBe('13');
    expect(modulus(s)).toBeLessThanOrEqual(modulus(z) + modulus(w));
  });
  it('anything else numerically', () => {
    const s = cxAdd(cxRect(q(1), q(0)), cxPolar(realQ(q(1)), turnQ(q(1, 6))));
    expect(s.rect).toBeNull();
    expect(s.re).toBeCloseTo(1.5, 15);
  });
});

describe('powers', () => {
  it('1.08∠24°: w¹⁵ is exactly on the positive real axis, at (27/25)¹⁵ ≈ 3.172', () => {
    const w = cxPolar(realQ(q(27, 25)), turnQ(q(1, 15)));
    const p = cxPow(w, 15);
    expect(txt(p.turn)).toBe('0');
    expect(p.rect![1].isZero()).toBe(true);
    expect(p.rect![0].eq(q(27, 25).pow(15))).toBe(true);
    expect(p.re).toBeCloseTo(3.172, 3);
  });
  it('1∠72°: w⁵ = 1 exactly; w¹ … w⁴ are not 1', () => {
    const w = cxPolar(realQ(q(1)), turnQ(q(1, 5)));
    expect(rectTxt(cxPow(w, 5))).toBe('1 0');
    for (let k = 1; k < 5; k++) expect(cxPow(w, k).rect).toBeNull();
  });
  it('w⁰ = 1, also for w = 0', () => {
    expect(rectTxt(cxPow(ZERO, 0))).toBe('1 0');
    expect(rectTxt(cxPow(ZERO, 3))).toBe('0 0');
    expect(rectTxt(cxPow(ONE, 7))).toBe('1 0');
  });
  it('(1 + i)⁸ = 16', () => {
    expect(rectTxt(cxPow(cxRect(q(1), q(1)), 8))).toBe('16 0');
  });
});

describe('surds', () => {
  it('√q = s√t', () => {
    const f = (x: Q) => { const r = surd(x); return `${r.s}√${r.t}`; };
    expect(f(q(10))).toBe('1√10');
    expect(f(q(8))).toBe('2√2');
    expect(f(q(45, 4))).toBe('3/2√5');
    expect(f(q(9, 4))).toBe('3/2√1');
    expect(f(q(1, 2))).toBe('1/2√2');
    expect(f(q(0))).toBe('0√1');
  });
});

describe('argDeg', () => {
  it('exact where the turn is, numeric elsewhere, none at 0', () => {
    expect(argDeg(cxRect(q(1), q(1)))!.exact!.toString()).toBe('45');
    expect(argDeg(cxRect(q(-1), q(0)))!.exact!.toString()).toBe('180');
    const a = argDeg(cxRect(q(2), q(1)))!;
    expect(a.exact).toBeNull();
    expect(a.value).toBeCloseTo(26.565, 3);
    expect(argDeg(ZERO)).toBeNull();
  });
});

describe('exactParts', () => {
  it('e^{iπ/3} = 1/2 + (√3/2)i: squares 1/4 and 3/4', () => {
    const p = exactParts(cxPolar(realQ(q(1)), turnQ(q(1, 6))))!;
    expect(p.map((x) => `${x.sign}:${x.sq}`)).toEqual(['1:1/4', '1:3/4']);
  });
  it('2∠−120° = −1 − √3 i', () => {
    const p = exactParts(cxPolar(realQ(q(2)), turnQ(q(-1, 3))))!;
    expect(p.map((x) => `${x.sign}:${x.sq}`)).toEqual(['-1:1', '-1:3']);
  });
  it('rational parts; none at 72°', () => {
    expect(exactParts(cxRect(q(-2), q(0)))!.map((x) => `${x.sign}:${x.sq}`)).toEqual(['-1:4', '0:0']);
    expect(exactParts(cxPolar(realQ(q(1)), turnQ(q(1, 5))))).toBeNull();
  });
});

describe('sums that cancel', () => {
  it('e^{iπ/3} + e^{−iπ/3} = 1 exactly', () => {
    const s = cxAdd(cxPolar(realQ(q(1)), turnQ(q(1, 6))), cxPolar(realQ(q(1)), turnQ(q(-1, 6))));
    expect(rectTxt(s)).toBe('1 0');
  });
  it('0 + w keeps every exact part of w', () => {
    const w = cxPolar(realQ(q(1)), turnQ(q(1, 5)));
    expect(cxAdd(ZERO, w)).toBe(w);
  });
  it('the same exact turn is kept', () => {
    const s = cxAdd(cxPolar(realQ(q(1)), turnQ(q(1, 5))), cxPolar(realQ(q(2)), turnQ(q(1, 5))));
    expect(txt(s.turn)).toBe('1/5');
    expect(modulus(s)).toBeCloseTo(3, 14);
  });
});

describe('tiny values and provable equality (Codex implementation review 6, 7)', () => {
  it('a non-zero exact square whose double underflows is not 0, and keeps its direction exactly', () => {
    const tiny = cxPolar({ v: 0, sq: Q.of(1n, 10n ** 400n) }, turnQ(q(1, 9)));
    expect(isZero(tiny)).toBe(false);
    expect(argDeg(tiny)!.exact!.toString()).toBe('40');
  });
  it('rescaling to an exact length does not underflow', () => {
    const p = cxMul(cxNum(2, 1), cxPolar(realQ(q(1, 10 ** 15)), turnQ(q(0))));
    expect(p.re).toBeGreaterThan(0);
  });
  it('equal only when provably so', () => {
    expect(provablyEqual(cxRect(q(1), q(0)), ONE)).toBe(true);
    expect(provablyEqual(cxPolar(realQ(q(1)), turnQ(q(1, 4))), I)).toBe(true);
    expect(provablyEqual(cxNum(1, 0), ONE)).toBe(false);
    expect(provablyEqual(cxRect(q(10000000001, 10000000000), q(0)), ONE)).toBe(false);
  });
});
