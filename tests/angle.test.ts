// Angles as the trigonometry pages hold them (src/lib/math/angle.ts, lifted from C01): the two states, a range checked
// exactly, the declared table, the π and degree texts, typing within a page's range. C01's own tests cover the same
// functions through its page (tests/unit-circle-math.test.ts on its branch).
import { describe, expect, test } from 'vitest';
import { Q } from '../src/lib/math/rational';
import {
  angleIn, degText, editAngle, exactAngle, exactQ, numericAngle, piRange, piText, plusQ, radText, readAngle, sameAngle, snapK, specialK, stepDegrees,
  stepSpecial, valText, valuesAt,
} from '../src/lib/math/angle';

describe('angle.ts', () => {
  test('a range is checked exactly for an exact angle, numerically for a numeric one', () => {
    const r = piRange(0, 2);
    expect(angleIn(exactQ(Q.of(2)), r)).toBe(true);
    expect(angleIn(exactQ(Q.of(2001, 1000)), r)).toBe(false);
    expect(angleIn(numericAngle(2 * Math.PI), r)).toBe(true);
    expect(angleIn(numericAngle(6.2832), r)).toBe(false);
    expect(angleIn(numericAngle(-1e-300), r)).toBe(false);
    expect(angleIn(exactQ(Q.of(-20)), piRange(-20, 20))).toBe(true);
  });
  test('plusQ: exact + π/2 is exact; numeric stays numeric', () => {
    expect(plusQ(exactAngle(2), Q.of(1, 2)).q?.eq(Q.of(2, 3))).toBe(true);
    expect(plusQ(numericAngle(1), Q.of(1, 2)).q).toBeNull();
    expect(plusQ(numericAngle(1), Q.of(1, 2)).theta).toBeCloseTo(1 + Math.PI / 2, 15);
  });
  test('the declared table: closed forms only at the special angles', () => {
    expect(valText(valuesAt(exactAngle(8)).cos)).toBe('−1/2');
    expect(valuesAt(exactAngle(6)).tan.kind).toBe('undef');
    expect(valuesAt(numericAngle(Math.PI / 2)).tan.kind).toBe('num');
    expect(specialK(exactQ(Q.of(61, 360)))).toBeNull();
    expect(specialK(exactAngle(26))).toBe(26);
  });
  test('texts: π forms, degrees exactly when they terminate, ≈ otherwise', () => {
    expect(piText(Q.of(-11, 6))).toBe('−11π/6');
    expect(piText(Q.of(2, 3), true)).toBe('2pi/3');
    expect(radText(numericAngle(1))).toEqual({ rel: '=', text: '1' });
    expect(degText(exactAngle(8))).toEqual({ rel: '=', text: '120°' });
    expect(degText(numericAngle(1))).toEqual({ rel: '≈', text: '57.2958°' });
  });
  test('typing within a page\'s range: π forms and degrees exact, decimals numeric; out of range refused', () => {
    const r = piRange(0, 2);
    const a = readAngle('120°', r);
    expect(a.ok && a.angle.q?.eq(Q.of(2, 3))).toBe(true);
    const b = readAngle('2.0944', r);
    expect(b.ok && b.angle.q === null).toBe(true);
    expect(readAngle('-30°', r)).toEqual({ ok: false, reason: 'range' });
    expect(readAngle('-30°', piRange(-20, 20)).ok).toBe(true);
    expect(readAngle('', r)).toEqual({ ok: false, reason: 'nan' });
    expect(editAngle(exactAngle(4))).toBe('π/3');
    expect(editAngle(numericAngle(0.5235987755982988))).toBe('0.5235987755982988');
  });
  test('moving: degree steps keep exactness, special steps and the catch', () => {
    expect(sameAngle(stepDegrees(stepDegrees(exactAngle(8), 10), -10), exactAngle(8))).toBe(true);
    expect(stepSpecial(exactQ(Q.of(31, 180)), 1)).toBe(3);
    expect(stepSpecial(exactAngle(0), -1)).toBe(-2);
    expect(snapK(0.5236, null, 0.01, 0.02)).toBe(2);
    expect(snapK(0.6, null, 0.01, 0.02)).toBeNull();
    expect(snapK(0.54, 2, 0.01, 0.02)).toBe(2);
  });
});
