// The discrete Fourier transform, for inverting characteristic functions on a grid (the central limit theorem page's
// numeric densities: src/lib/math/sums.ts). Iterative radix-2, in place on two Float64Arrays (real and imaginary
// parts), length a power of two. Forward: X_k = Σ_j x_j e^{−2πi jk/N}; inverse: x_j = (1/N) Σ_k X_k e^{+2πi jk/N}, so
// a forward then inverse transform gives the input back. Twiddles come from cos / sin of exact multiples of 2π/N (not
// from repeated multiplication), so their error does not grow with N.

/** The smallest power of 2 that is ≥ n (1 for n ≤ 1). */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/** In-place FFT of (re, im); `inverse` transforms back and divides by N. RangeError unless the length is a power of 2
 *  and both arrays have it. */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const N = re.length;
  if (im.length !== N) throw new RangeError('fft: real and imaginary parts must have the same length');
  if (N < 1 || (N & (N - 1)) !== 0) throw new RangeError('fft: the length must be a power of 2');
  if (N === 1) return;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // twiddles e^{∓2πik/N} for k < N/2, each from its own angle
  const half = N >> 1, sign = inverse ? 1 : -1;
  const wr = new Float64Array(half), wi = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    const a = (2 * Math.PI * k) / N;
    wr[k] = Math.cos(a);
    wi[k] = sign * Math.sin(a);
  }
  for (let len = 2; len <= N; len <<= 1) {
    const h = len >> 1, stride = N / len;
    for (let s = 0; s < N; s += len) {
      for (let k = 0; k < h; k++) {
        const c = wr[k * stride], d = wi[k * stride];
        const a = s + k, b = a + h;
        const xr = re[b] * c - im[b] * d, xi = re[b] * d + im[b] * c;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
  if (inverse) for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
}
