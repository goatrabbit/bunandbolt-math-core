// The shared reading of typed numbers (src/lib/ui/typed.ts, design spec §8.0.1).
import { describe, expect, test } from 'vitest';
import { editText, editTextQ, parseTyped, piMultiple, piRational, readTyped, readTypedQ, typedDegrees, typedDegreesQ } from '../src/lib/ui/typed';
import { parseExact, Q } from '../src/lib/math/rational';

const PI = Math.PI;

describe('decimals', () => {
  test('plain, signed, leading and trailing points, scientific', () => {
    expect(parseTyped('1')).toBe(1);
    expect(parseTyped(' 2.5 ')).toBe(2.5);
    expect(parseTyped('−2.5')).toBe(-2.5);
    expect(parseTyped('–3')).toBe(-3);
    expect(parseTyped('+4')).toBe(4);
    expect(parseTyped('.5')).toBe(0.5);
    expect(parseTyped('3.')).toBe(3);
    expect(parseTyped('1e-6')).toBe(1e-6);
    expect(parseTyped('2.5E3')).toBe(2500);
  });
  test('commas: one is a decimal point, groups of three are thousands', () => {
    expect(parseTyped('1,5')).toBe(1.5);
    expect(parseTyped('31,536,000')).toBe(31_536_000);
    expect(parseTyped('1,000')).toBe(1000);
    expect(parseTyped('12,345.6')).toBe(12345.6);
    expect(parseTyped('1,5,6')).toBeNaN();
    // per number, not per text (Codex review: "1,000π" was read as π)
    expect(parseTyped('1,000π')).toBe(1000 * Math.PI);
    expect(parseTyped('1,000/2')).toBe(500);
    expect(parseTyped('1,5π')).toBe(1.5 * Math.PI);
    expect(parseTyped('1,0,0π')).toBeNaN();
  });
  test('full-width characters from an IME', () => {
    expect(parseTyped('１２．５')).toBe(12.5);
    expect(parseTyped('－３')).toBe(-3);
    expect(parseTyped('１／３')).toBe(1 / 3);
  });
  test('−0 is 0', () => {
    expect(Object.is(parseTyped('-0'), 0)).toBe(true);
  });
});

describe('fractions, π, roots, e', () => {
  test('fractions', () => {
    expect(parseTyped('1/3')).toBe(1 / 3);
    expect(parseTyped('-7/2')).toBe(-3.5);
    expect(parseTyped('1/0')).toBeNaN();
  });
  test('multiples and fractions of π, written the way the pages print them', () => {
    expect(parseTyped('π')).toBe(PI);
    expect(parseTyped('pi')).toBe(PI);
    expect(parseTyped('PI')).toBe(PI);
    expect(parseTyped('2π')).toBe(2 * PI);
    expect(parseTyped('−π/2')).toBe(-PI / 2);
    expect(parseTyped('3π/4')).toBe((3 * PI) / 4);
    expect(parseTyped('0.5π')).toBe(0.5 * PI);
    expect(parseTyped('2*pi')).toBe(2 * PI);
    expect(parseTyped('π / 6')).toBe(PI / 6);
  });
  test('square roots', () => {
    expect(parseTyped('√3')).toBe(Math.sqrt(3));
    expect(parseTyped('−√2')).toBe(-Math.sqrt(2));
    expect(parseTyped('2√3')).toBe(2 * Math.sqrt(3));
    expect(parseTyped('√3/2')).toBe(Math.sqrt(3) / 2);
  });
  test('e', () => {
    expect(parseTyped('e')).toBe(Math.E);
    expect(parseTyped('-e')).toBe(-Math.E);
    expect(parseTyped('2e')).toBeNaN();
  });
});

describe('degrees (2026-10-07, C01): an angle typed in degrees is read in radians', () => {
  test('° and deg, signed, decimal', () => {
    expect(parseTyped('180°')).toBe(PI);
    expect(parseTyped('90deg')).toBe(PI / 2);
    expect(parseTyped('-30°')).toBe(-PI / 6);
    expect(parseTyped('−30 °')).toBe(-PI / 6);
    expect(parseTyped('120DEG')).toBe((2 * PI) / 3);
    expect(parseTyped('0°')).toBe(0);
    expect(parseTyped('３６０°')).toBe(2 * PI);
    expect(parseTyped('57.29577951308232°')).toBeCloseTo(1, 15);
  });
  test('the same number as the π form, so a field shows π forms for them while editing', () => {
    for (const [deg, pi] of [['120°', '2π/3'], ['22.5°', 'π/8'], ['330°', '11π/6'], ['-3600°', '-20π'], ['135°', '3π/4'], ['15°', 'π/12']]) {
      expect(parseTyped(deg), deg).toBe(parseTyped(pi));
    }
    expect(editText(parseTyped('120°'))).toBe('2π/3');
  });
  test('refused: a degree sign alone, twice, or on a form it does not belong to', () => {
    for (const t of ['°', 'deg', '120°°', '1/2°', 'π°', '√3°', '120 degrees', '120º', 'e°']) expect(parseTyped(t), t).toBeNaN();
    expect(readTyped('1e-999°')).toEqual({ ok: false, reason: 'underflow' });
  });
  test('typedDegrees gives the degrees typed, piMultiple their units of π', () => {
    expect(typedDegrees('120°')).toBe(120);
    expect(typedDegrees('-22.5deg')).toBe(-22.5);
    expect(typedDegrees('2π/3')).toBeNull();
    expect(typedDegrees('2.0944')).toBeNull();
    expect(piMultiple('30°')).toBe(1 / 6);
    expect(piMultiple('120°')).toBe(2 / 3);
    expect(piMultiple('-180deg')).toBe(-1);
  });
  test('piRational: π forms and degrees as exact multiples of π, decided from the text', () => {
    const r = (t: string) => piRational(t)?.toString() ?? null;
    expect(r('2π/3')).toBe('2/3');
    expect(r('-11pi/6')).toBe('-11/6');
    expect(r('0.5π')).toBe('1/2');
    expect(r('π/1.5')).toBe('2/3');
    expect(r('30.5°')).toBe('61/360');
    expect(r('30.0°')).toBe('1/6');
    expect(r('-3600°')).toBe('-20');
    expect(r('0°')).toBe('0');
    for (const t of ['0.5236', '0.5235987755982988', '√3', 'e', '1/2', 'abc', '']) expect(r(t), t).toBeNull();
  });
  test('typedDegreesQ: degrees exactly, π forms turned into degrees, roots refused', () => {
    const d = (t: string) => { const r = typedDegreesQ(t); return r.ok ? r.deg.toString() : r.reason; };
    expect(d('40')).toBe('40');
    expect(d('37.25')).toBe('149/4');
    expect(d('200/3')).toBe('200/3');
    expect(d('40°')).toBe('40');
    expect(d('-22.5deg')).toBe('-45/2');
    expect(d('π/4')).toBe('45');
    expect(d('2pi/3')).toBe('120');
    expect(d('√3')).toBe('irrational');
    expect(d('e')).toBe('irrational');
    expect(d('abc')).toBe('nan');
    expect(d('')).toBe('nan');
  });
  test('what was read before reads the same', () => {
    expect(parseTyped('2.0944')).toBe(2.0944);
    expect(parseTyped('2π/3')).toBe((2 * PI) / 3);
    expect(piMultiple('0.5')).toBeNull();
  });
});

describe('refused, never guessed', () => {
  test('blank, junk and forms Number() would accept', () => {
    expect(readTyped('')).toEqual({ ok: false, reason: 'empty' });
    expect(readTyped('  ')).toEqual({ ok: false, reason: 'empty' });
    for (const t of ['abc', '1..2', '0x10', '0b1', 'Infinity', '1e', '--1', '2π2', '√', '1/2/3']) expect(parseTyped(t), t).toBeNaN();
    // superscripts and vulgar fractions are not guessed at (Codex review: "2²" read as 22, "1½" as 5.5)
    for (const t of ['2²', '1½', '½', '10⁻³', '①']) expect(parseTyped(t), t).toBeNaN();
  });
  test('overflow and underflow', () => {
    expect(readTyped('1e999')).toEqual({ ok: false, reason: 'nan' });
    expect(readTyped('1e-999')).toEqual({ ok: false, reason: 'underflow' });
    expect(readTyped('-2e-400')).toEqual({ ok: false, reason: 'underflow' });
    expect(readTyped('1e-400π')).toEqual({ ok: false, reason: 'underflow' });
    expect(readTyped('1e-999√3')).toEqual({ ok: false, reason: 'underflow' });
    expect(parseTyped('√0')).toBe(0);
    expect(parseTyped('0π')).toBe(0);
    expect(parseTyped('0e5')).toBe(0);
  });
});

describe('editText: what a field shows while editing reads back as exactly the value', () => {
  test('round trip for plain, π, √ and awkward values', () => {
    const vs = [0, 1, -2.5, 0.1, 0.1 + 0.2, 1 / 3, Math.PI, -Math.PI / 2, (3 * Math.PI) / 4, 2 * Math.PI, Math.sqrt(3), -Math.sqrt(2) / 2,
      Math.E, 1e-7, 1.234567890123456, 31_536_000, 1e21, -6.5, 5e-324, 1.7976931348623157e308];
    for (const v of vs) expect(parseTyped(editText(v)), `${v} → "${editText(v)}"`).toBe(v);
  });
  test('short forms where they are exact', () => {
    expect(editText(Math.PI)).toBe('π');
    expect(editText(-Math.PI / 2)).toBe('-π/2');
    expect(editText(Math.sqrt(3))).toBe('√3');
    expect(editText(2.5)).toBe('2.5');
    expect(editText(0.1 + 0.2)).toBe('0.30000000000000004');
  });
  test('editText guesses π forms from the float; editTextQ writes from the state and never guesses', () => {
    const near = (2 * Math.PI) / 3; // a numeric angle that is the double nearest 2π/3
    expect(editText(near)).toBe('2π/3'); // read back with piRational, this would turn exact
    expect(editTextQ(near, null)).toBe('2.0943951023931953');
    expect(piRational(editTextQ(near, null))).toBeNull();
    expect(editTextQ(near, Q.of(2, 3), { pi: true })).toBe('2π/3');
    for (const q of [Q.of(2, 3), Q.of(-11, 6), Q.of(20), Q.of(1), Q.of(-1), Q.ZERO, Q.of(61, 360)]) {
      const t = editTextQ(q.toNumber() * Math.PI, q, { pi: true });
      expect(piRational(t)?.eq(q) ?? q.isZero(), t).toBe(true);
      expect(parseTyped(t), t).toBe(q.isZero() ? 0 : (Number(q.n) * Math.PI) / Number(q.d));
    }
    expect(editTextQ(-Math.PI / 2, Q.of(-1, 2), { pi: true })).toBe('-π/2');
    for (const q of [Q.of(2, 3), Q.of(-7, 2), Q.of(3), Q.of(1, 10)]) expect(parseExact(editTextQ(q.toNumber(), q))?.eq(q)).toBe(true);
    expect(editTextQ(0.1, Q.of(1, 10))).toBe('1/10');
    for (const v of [1, -2.5, 0.1 + 0.2, 1e-7, 1e21, -0]) expect(parseTyped(editTextQ(v, null)), String(v)).toBe(v + 0);
  });
  test('piMultiple reads units of π exactly', () => {
    expect(piMultiple('π/6')).toBe(1 / 6);
    expect(piMultiple('-2π')).toBe(-2);
    expect(piMultiple('0.5')).toBeNull();
  });
});

describe('readTypedQ: the same text, read exactly', () => {
  const q = (t: string) => { const r = readTypedQ(t); return r.ok ? r.q.toString() : r.reason; };
  test('decimals, scientific notation and fractions are the rationals they name', () => {
    expect(q('4.1')).toBe('41/10');
    expect(q('−0.1')).toBe('-1/10');
    expect(q('1e-6')).toBe('1/1000000');
    expect(q('2.5E-7')).toBe('1/4000000');
    expect(q('.5')).toBe('1/2');
    expect(q('3.')).toBe('3');
    expect(q('1/3')).toBe('1/3');
    expect(q('0.5/2')).toBe('1/4');
    expect(q('1,5')).toBe('3/2');
    expect(q('31,536,000')).toBe('31536000');
    expect(q('１２．５')).toBe('25/2');
    expect(q('0.12345678')).toBe('6172839/50000000'); // no six-place limit
  });
  test('4.1 − 0.1 is exactly 4 (the binary doubles are not)', () => {
    const a = readTypedQ('4.1'), b = readTypedQ('0.1');
    expect(a.ok && b.ok && a.q.sub(b.q).eq(Q.of(4))).toBe(true);
    expect(4.1 - 0.1).not.toBe(4);
  });
  test('π, √ and e are refused as irrational; junk as readTyped refuses it', () => {
    expect(q('π/2')).toBe('irrational');
    expect(q('√2')).toBe('irrational');
    expect(q('e')).toBe('irrational');
    expect(q('abc')).toBe('nan');
    expect(q('')).toBe('empty');
    expect(q('1e-999')).toBe('underflow');
    expect(q('1/0')).toBe('nan');
  });
  test('the value is the nearest double', () => {
    const r = readTypedQ('0.1');
    expect(r.ok && r.value).toBe(0.1);
  });
});
