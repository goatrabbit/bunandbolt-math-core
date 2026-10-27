// Example 1: a typed 0.1 is one tenth.
//
// A reader who types 0.1 means the decimal 1/10. The double nearest to it is 3602879701896397 / 2⁵⁵, a little more than
// 0.1, which is why 0.1 + 0.2 === 0.3 is false in JavaScript. The pages read typed text with parseExact and make every
// decision (equal or not, zero or not, which case) from that exact value. Q.fromDouble takes a double's exact binary
// value instead: for numbers a page computed or a drag produced, which never were decimals.
import { describe, expect, it } from 'vitest';
import { parseExact, Q } from '../src/lib/math/rational';

describe('a typed 0.1', () => {
  it('is read as 1/10, not as the nearest double', () => {
    const typed = parseExact('0.1')!;
    expect([typed.n, typed.d]).toEqual([1n, 10n]);
    const double = Q.fromDouble(0.1);
    expect([double.n, double.d]).toEqual([3602879701896397n, 2n ** 55n]);
  });

  it('adds up exactly', () => {
    expect(0.1 + 0.2 === 0.3).toBe(false);
    expect(parseExact('0.1')!.add(parseExact('0.2')!).eq(parseExact('0.3')!)).toBe(true);
  });
});
