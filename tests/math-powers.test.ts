// src/lib/math/powers.ts: exact positive reals e^a·∏gᵢ^rᵢ and exact rational logarithms (2026-10-07, A12).
import { describe, expect, it } from 'vitest';
import { Pow, coprimeBase } from '../src/lib/math/powers';
import { Q } from '../src/lib/math/rational';
import { readTyped, typedExact } from '../src/lib/ui/typed';

const P = (n: number, d = 1) => Pow.of(n, d);
const root = (n: number) => P(n).pow(Q.of(1, 2));
const str = (q: Q | null) => (q ? q.toString() : null);

describe('coprimeBase', () => {
  it('refines into pairwise coprime factors that rebuild every input', () => {
    const inputs = [12n, 18n, 1000n, 35n, 49n, 1n];
    const S = coprimeBase(inputs);
    for (let i = 0; i < S.length; i++) for (let j = i + 1; j < S.length; j++) {
      let a = S[i], b = S[j];
      while (b) [a, b] = [b, a % b];
      expect(a).toBe(1n);
    }
    for (const n of inputs.filter((x) => x > 1n)) {
      let rest = n;
      for (const s of S) while (rest % s === 0n) rest /= s;
      expect(rest).toBe(1n);
    }
  });
  it('drops 1 and duplicates', () => {
    expect(coprimeBase([1n, 6n, 6n])).toEqual([6n]);
    expect(coprimeBase([6n, 4n])).toEqual([2n, 3n]);
  });
});

describe('Pow arithmetic', () => {
  it('multiplies, divides and takes rational powers exactly', () => {
    expect(P(6).mul(P(1, 2)).eq(P(3))).toBe(true);
    expect(P(12).div(P(18)).eq(P(2, 3))).toBe(true);
    expect(root(2).mul(root(2)).eq(P(2))).toBe(true);
    expect(root(8).eq(P(2).pow(Q.of(3, 2)))).toBe(true);
    expect(P(10).powInt(12).toQ()!.eq(Q.of(10n ** 12n))).toBe(true);
    expect(Pow.E.mul(Pow.E).eq(Pow.E.pow(Q.of(2)))).toBe(true);
    expect(Pow.E.eq(P(3))).toBe(false);
    expect(P(1).isOne()).toBe(true);
    expect(P(4).mul(P(1, 4)).isOne()).toBe(true);
  });
  it('equality refines different bases (6 = 2·3)', () => {
    expect(P(6).eq(P(2).mul(P(3)))).toBe(true);
    expect(P(36).pow(Q.of(1, 2)).eq(P(6))).toBe(true);
  });
  it('toNumber agrees with the double arithmetic', () => {
    expect(root(2).toNumber()).toBeCloseTo(Math.SQRT2, 15);
    expect(P(2).mul(root(2)).toNumber()).toBeCloseTo(2 * Math.SQRT2, 14);
    expect(Pow.E.toNumber()).toBe(Math.E);
    expect(P(3).powInt(12).toNumber()).toBe(531441);
    expect(P(1, 1024).toNumber()).toBe(2 ** -10);
    expect(P(137, 100).toNumber()).toBe(1.37);
  });
  it('ln keeps the digits of a base near 1', () => {
    expect(P(1001, 1000).ln()).toBe(Math.log1p(0.001));
    expect(P(8).ln()).toBe(Math.log(8));
    expect(Pow.E.pow(Q.of(3)).ln()).toBe(3);
  });
  it('refuses zero and negatives', () => {
    expect(() => Pow.ofQ(Q.ZERO)).toThrow();
    expect(() => Pow.ofQ(Q.of(-2))).toThrow();
  });
});

describe('logIn: rational exactly when the exponent vectors are proportional', () => {
  it('finds the rational logarithms', () => {
    expect(str(P(8).logIn(P(2)))).toBe('3');
    expect(str(P(1000).logIn(P(10)))).toBe('3'); // the double quotient is 2.9999999999999996
    expect(Math.log(1000) / Math.log(10)).not.toBe(3);
    expect(str(root(2).logIn(P(2)))).toBe('1/2');
    expect(str(P(8).logIn(P(4)))).toBe('3/2');
    expect(str(root(3).logIn(P(9)))).toBe('1/4');
    expect(str(P(8).logIn(P(1, 2)))).toBe('-3');
    expect(str(P(1, 10).logIn(P(10)))).toBe('-1');
    expect(str(P(10).logIn(P(1, 10)))).toBe('-1');
    expect(str(P(1).logIn(P(7)))).toBe('0');
    expect(str(Pow.E.pow(Q.of(2)).logIn(Pow.E))).toBe('2');
    expect(str(P(9, 4).logIn(P(3, 2)))).toBe('2');
    expect(str(P(12).logIn(P(144)))).toBe('1/2');
    expect(str(P(2).mul(root(2)).logIn(P(2)))).toBe('3/2');
  });
  it('says null when the logarithm is irrational', () => {
    expect(P(3).logIn(P(2))).toBeNull();
    expect(P(10).logIn(P(2))).toBeNull();
    expect(P(6).logIn(P(2))).toBeNull();
    expect(Pow.E.logIn(P(2))).toBeNull();
    expect(P(2).logIn(Pow.E)).toBeNull();
    expect(P(2).mul(Pow.E).logIn(Pow.E)).toBeNull();
    expect(P(12).logIn(P(18))).toBeNull();
  });
  it('refuses base 1', () => {
    expect(() => P(2).logIn(P(1))).toThrow();
    expect(() => P(2).logIn(P(5).div(P(5)))).toThrow();
  });
  it('agrees with the doubles wherever it is rational', () => {
    const vals = [P(2), P(3), P(10), P(1, 2), root(2), P(4), P(27), P(1, 9), Pow.E, P(6), P(36), root(6)];
    for (const x of vals) for (const b of vals) {
      if (b.isOne()) continue;
      const q = x.logIn(b);
      if (q) expect(q.toNumber()).toBeCloseTo(x.ln() / b.ln(), 12);
    }
  });
});

describe('Codex implementation review 4, 6: perfect powers, sums of logarithms, digits near 1', () => {
  it('sees a root of a perfect square as rational', () => {
    expect(Pow.of(1002001, 1000000).pow(Q.of(1, 2)).toQ()?.toString()).toBe('1001/1000');
    expect(Pow.of(64).pow(Q.of(1, 3)).toQ()?.toString()).toBe('4');
    expect(Pow.of(1, 1000000).pow(Q.of(1, 2)).toQ()?.toString()).toBe('1/1000');
    expect(Pow.of(8).pow(Q.of(1, 2)).toQ()).toBeNull();
  });
  it('a product whose parts overflow on their own', () => {
    const v = Pow.of(2).powInt(2000).mul(Pow.E.pow(Q.of(-1386))).toNumber();
    expect(v).toBeCloseTo(Math.exp(2000 * Math.LN2 - 1386), 9);
  });
  it('ln of a root just above 1 keeps its digits', () => {
    const x = Pow.ofQ(Q.of(10n ** 20n + 1n, 10n ** 20n)).pow(Q.of(1, 2));
    expect(x.ln()).toBeCloseTo(5e-21, 30);
    expect(x.ln() / 5e-21).toBeCloseTo(1, 12);
  });
});

describe('surd: q·√r·eᵏ for writing a value out', () => {
  it('extracts the square part', () => {
    const s = (p: Pow) => { const v = p.surd()!; return `${v.q}|${v.r}|${v.e}`; };
    expect(s(root(8))).toBe('2|2|0');
    expect(s(root(12))).toBe('2|3|0');
    expect(s(P(1, 2).pow(Q.of(1, 2)))).toBe('1/2|2|0');
    expect(s(P(27))).toBe('27|1|0');
    expect(s(Pow.E.pow(Q.of(3)).mul(P(2)))).toBe('2|1|3');
    expect(P(2).pow(Q.of(1, 3)).surd()).toBeNull();
  });
});

describe('typedExact reads what readTyped reads, exactly', () => {
  const cases: [string, string, string | null, 0 | 1][] = [
    ['2', '2', null, 0], ['0.1', '1/10', null, 0], ['1e-3', '1/1000', null, 0], ['1,5', '3/2', null, 0],
    ['3/4', '3/4', null, 0], ['√2', '1', '2', 0], ['2√3/4', '1/2', '3', 0], ['√0.5', '1', '1/2', 0],
    ['e', '1', null, 1], ['−2.5', '-5/2', null, 0], ['１２', '12', null, 0],
  ];
  for (const [text, coef, rootText, e] of cases) {
    it(text, () => {
      const r = typedExact(text)!;
      expect(r.coef.toString()).toBe(coef);
      expect(r.root ? r.root.toString() : null).toBe(rootText);
      expect(r.e).toBe(e);
      const v = r.coef.toNumber() * (r.root ? Math.sqrt(r.root.toNumber()) : 1) * (r.e ? Math.E : 1);
      const t = readTyped(text);
      expect(t.ok && Math.abs(t.value - v) <= 1e-15 * Math.abs(v)).toBe(true);
    });
  }
  it('refuses π, degrees and junk', () => {
    for (const t of ['π', '2π/3', '30°', 'abc', '', '1e999', '0x10']) expect(typedExact(t)).toBeNull();
  });
});
