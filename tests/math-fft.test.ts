import { describe, expect, it } from 'vitest';
import { fft, nextPow2 } from '../src/lib/math/fft';
import { mulberry32 } from '../src/lib/math/rng';

// The FFT inverts characteristic functions for the central limit theorem page (src/lib/math/sums.ts): it must agree
// with the plain DFT sum, come back to its input, and refuse lengths it cannot do.

/** The DFT by its definition, O(N²): the reference. */
function dft(re: number[], im: number[], inverse = false): [number[], number[]] {
  const N = re.length, sign = inverse ? 1 : -1, or: number[] = [], oi: number[] = [];
  for (let k = 0; k < N; k++) {
    let sr = 0, si = 0;
    for (let j = 0; j < N; j++) {
      const a = (sign * 2 * Math.PI * j * k) / N, c = Math.cos(a), s = Math.sin(a);
      sr += re[j] * c - im[j] * s;
      si += re[j] * s + im[j] * c;
    }
    or.push(inverse ? sr / N : sr);
    oi.push(inverse ? si / N : si);
  }
  return [or, oi];
}

describe('fft', () => {
  it('matches the DFT sum on random data (N = 1 … 64)', () => {
    const r = mulberry32(3);
    for (const N of [1, 2, 4, 8, 64]) {
      const re = Array.from({ length: N }, () => r() - 0.5), im = Array.from({ length: N }, () => r() - 0.5);
      const [er, ei] = dft(re, im);
      const a = Float64Array.from(re), b = Float64Array.from(im);
      fft(a, b);
      for (let k = 0; k < N; k++) {
        expect(a[k]).toBeCloseTo(er[k], 12);
        expect(b[k]).toBeCloseTo(ei[k], 12);
      }
    }
  });
  it('a known transform: a unit impulse is flat, a cosine is two spikes', () => {
    const a = new Float64Array(8), b = new Float64Array(8);
    a[0] = 1;
    fft(a, b);
    expect([...a]).toEqual(Array(8).fill(1));
    const c = Float64Array.from({ length: 16 }, (_, j) => Math.cos((2 * Math.PI * 3 * j) / 16)), d = new Float64Array(16);
    fft(c, d);
    for (let k = 0; k < 16; k++) expect(Math.hypot(c[k], d[k])).toBeCloseTo(k === 3 || k === 13 ? 8 : 0, 12);
  });
  it('forward then inverse gives the input back (N = 4096)', () => {
    const r = mulberry32(11), N = 4096;
    const re = Float64Array.from({ length: N }, () => r()), im = Float64Array.from({ length: N }, () => r());
    const a = re.slice(), b = im.slice();
    fft(a, b);
    fft(a, b, true);
    let worst = 0;
    for (let i = 0; i < N; i++) worst = Math.max(worst, Math.abs(a[i] - re[i]), Math.abs(b[i] - im[i]));
    expect(worst).toBeLessThan(1e-13);
  });
  it('refuses a length that is not a power of 2, and mismatched parts', () => {
    expect(() => fft(new Float64Array(6), new Float64Array(6))).toThrow(RangeError);
    expect(() => fft(new Float64Array(0), new Float64Array(0))).toThrow(RangeError);
    expect(() => fft(new Float64Array(8), new Float64Array(4))).toThrow(RangeError);
  });
  it('nextPow2', () => {
    expect([0, 1, 2, 3, 5, 1024, 1025].map(nextPow2)).toEqual([1, 1, 2, 4, 8, 1024, 2048]);
  });
});
