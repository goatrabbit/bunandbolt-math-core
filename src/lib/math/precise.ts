// Rigorous enclosures of π and of logarithms, as exact rationals [lo, hi] with lo ≤ the true value ≤ hi, for the rare
// exact decision a double cannot make: G05 asks whether a typed damping ratio lies above or below the transcendental
// ζₙ = ln100/√(n²π² + ln²100), which only a value typed to some fourteen digits brings within a double's reach.
//
// Fixed point in BigInt: every series term is truncated (each truncation is counted in the error bound), and the tail
// after the last nonzero term is bounded by that term. π by Machin's formula 16·atan(1/5) − 4·atan(1/239); ln x by
// ln x = k·ln 2 + 2·atanh((m − 1)/(m + 1)) with x = 2ᵏ·m, m in [3/4, 3/2], and ln 2 = 2·atanh(1/3). The width of the
// enclosure is below 2^−bits.
import { Q } from './rational';

/** A fixed-point value v / 2^B with |error| ≤ err units. */
interface Fixed { v: bigint; err: bigint }

/** atan(1/k) · 2^B for an integer k ≥ 2: the alternating series, each power truncated. */
function atanInv(k: bigint, B: bigint): Fixed {
  const k2 = k * k;
  let p = (1n << B) / k, sum = 0n, n = 0n, terms = 0n;
  // p carries less than 2 units of accumulated truncation (each step divides the old error by k² ≥ 4 and adds < 1)
  while (p > 0n) {
    const t = p / (2n * n + 1n);
    sum += n % 2n === 0n ? t : -t;
    p /= k2;
    n++;
    terms++;
  }
  // per term: < 2 from p, < 1 from the division; the omitted tail is smaller than the first omitted term (< 2)
  return { v: sum, err: 3n * terms + 2n };
}

/** atanh(a/b) · 2^B for 0 ≤ a/b ≤ 1/3: the series of odd powers, each power truncated. */
function atanhRatio(a: bigint, b: bigint, B: bigint): Fixed {
  const a2 = a * a, b2 = b * b;
  let p = ((1n << B) * a) / b, sum = 0n, n = 0n, terms = 0n;
  while (p > 0n) {
    sum += p / (2n * n + 1n);
    p = (p * a2) / b2;
    n++;
    terms++;
  }
  // accumulated truncation in p stays below 1/(1 − 1/9) < 2 units; the tail is a geometric series below 2 units
  return { v: sum, err: 3n * terms + 4n };
}

const bounds = (f: Fixed, B: bigint): [Q, Q] => [Q.of(f.v - f.err, 1n << B), Q.of(f.v + f.err, 1n << B)];
const guard = (bits: number) => BigInt(Math.max(16, Math.ceil(bits)) + 24);

/** π enclosed in an interval narrower than 2^−bits. */
export function piBounds(bits = 200): [Q, Q] {
  const B = guard(bits);
  const a = atanInv(5n, B), b = atanInv(239n, B);
  return bounds({ v: 16n * a.v - 4n * b.v, err: 16n * a.err + 4n * b.err }, B);
}

/** ln x for a positive rational x, enclosed in an interval narrower than about 2^−bits (wider by the factor |k| for
 *  x far from 1, where x = 2ᵏ·m). */
export function lnBounds(x: Q, bits = 200): [Q, Q] {
  if (x.sign() <= 0) throw new RangeError('ln of a non-positive number');
  const B = guard(bits) + 8n;
  // x = 2^k · m with m in [3/4, 3/2]
  let k = x.ilog2();
  let m = x.div(k >= 0 ? Q.of(1n << BigInt(k)) : Q.of(1n, 1n << BigInt(-k)));
  while (m.cmp(Q.of(3, 2)) > 0) { m = m.div(Q.of(2)); k++; }
  while (m.cmp(Q.of(3, 4)) < 0) { m = m.mul(Q.of(2)); k--; }
  const r = m.sub(Q.ONE).div(m.add(Q.ONE)); // in [−1/7, 1/5]
  const neg = r.sign() < 0, ar = r.abs();
  const t = atanhRatio(ar.n, ar.d, B);
  const ln2 = atanhRatio(1n, 3n, B);
  const K = BigInt(k);
  const v = 2n * (neg ? -t.v : t.v) + 2n * K * ln2.v;
  const err = 2n * t.err + 2n * (K < 0n ? -K : K) * ln2.err;
  return bounds({ v, err }, B);
}
