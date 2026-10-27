// Exact rational numbers (BigInt numerator / denominator, always reduced, denominator > 0) for the pages that show
// exact values: the quadratic page's three forms, the linear-algebra pages' matrices and solutions (1/6, 2/3), the
// least-squares fits on small integer data. Generalised from src/scenes/quadratic-function/rational.ts (2026-10-07);
// that page keeps its own copy until it moves to this one.
//
// Only a handful of operations run per update, so BigInt costs nothing noticeable. `approximate` is the one inexact
// entry: it turns a double into the best rational with a bounded denominator and says so.
//
// Two ways in, for two kinds of number: `parseExact` reads what a reader typed ("0.1" is 1/10, "1/3" is 1/3) — every
// decision a page makes from typed values starts here; `Q.fromDouble` takes a double's exact binary value (the double
// 0.1 is 3602879701896397/2⁵⁵) — for values the page computed or a drag produced, which never were decimals.

const babs = (x: bigint) => (x < 0n ? -x : x);
function gcd(a: bigint, b: bigint): bigint {
  a = babs(a);
  b = babs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

export class Q {
  readonly n: bigint;
  readonly d: bigint;
  private constructor(n: bigint, d: bigint) {
    this.n = n;
    this.d = d;
  }
  static of(n: bigint | number, d: bigint | number = 1n): Q {
    let N = BigInt(n), D = BigInt(d);
    if (D === 0n) throw new RangeError('zero denominator');
    if (D < 0n) { N = -N; D = -D; }
    const g = gcd(N, D) || 1n;
    return new Q(N / g, D / g);
  }
  static readonly ZERO = Q.of(0);
  static readonly ONE = Q.of(1);
  /**
   * The exact value of a double: every finite double is a dyadic rational m / 2ᵏ. Doubling is exact, so this loses
   * nothing; use it to decide exact questions (is this determinant zero?) about the binary numbers a page holds. For
   * values the reader typed, use `parseExact` on the text instead: the double 0.1 is not 1/10.
   */
  static fromDouble(x: number): Q {
    if (!Number.isFinite(x)) throw new RangeError('not a finite number');
    let m = x, k = 0;
    while (!Number.isInteger(m)) {
      m *= 2;
      k++;
    }
    return Q.of(BigInt(m), 1n << BigInt(k));
  }

  add(o: Q) { return Q.of(this.n * o.d + o.n * this.d, this.d * o.d); }
  sub(o: Q) { return Q.of(this.n * o.d - o.n * this.d, this.d * o.d); }
  mul(o: Q) { return Q.of(this.n * o.n, this.d * o.d); }
  div(o: Q) {
    if (o.n === 0n) throw new RangeError('division by zero');
    return Q.of(this.n * o.d, this.d * o.n);
  }
  neg() { return new Q(-this.n, this.d); }
  abs() { return this.n < 0n ? this.neg() : this; }
  sign(): -1 | 0 | 1 { return this.n === 0n ? 0 : this.n < 0n ? -1 : 1; }
  isZero() { return this.n === 0n; }
  isInteger() { return this.d === 1n; }
  eq(o: Q) { return this.n === o.n && this.d === o.d; }
  cmp(o: Q): -1 | 0 | 1 {
    const x = this.n * o.d - o.n * this.d;
    return x === 0n ? 0 : x < 0n ? -1 : 1;
  }
  /** The nearest double (correctly rounded, half to even), for any size of numerator and denominator. */
  toNumber(): number {
    if (this.n === 0n) return 0;
    const neg = this.n < 0n, a = neg ? -this.n : this.n, d = this.d;
    const e = bitLen(a) - bitLen(d); // a/d lies in [2^(e−1), 2^(e+1))
    let v: number;
    if (e > 1025) v = Infinity;
    else if (e < -1080) v = 0; // below half the smallest subnormal
    else if (a << 1022n < d) {
      // subnormal: count units of 2^−1074 and round half to even by hand (q < 2^52, so Number(q) is exact)
      const num = a << 1074n;
      let q = num / d;
      const r2 = (num % d) * 2n;
      if (r2 > d || (r2 === d && (q & 1n) === 1n)) q += 1n;
      v = scale2(Number(q), -1074);
    } else {
      // an integer quotient of 55–56 bits with a sticky last bit: Number() then rounds it exactly once
      const s = 55 - e;
      const num = s >= 0 ? a << BigInt(s) : a, den = s >= 0 ? d : d << BigInt(-s);
      let q = num / den;
      if (num % den !== 0n) q |= 1n;
      v = scale2(Number(q), -s);
    }
    return neg ? -v : v;
  }
  /** ⌊log₂|q|⌋ or one less (from the bit lengths), for scaling by a power of two; −Infinity for 0. */
  ilog2(): number {
    return this.n === 0n ? -Infinity : bitLen(this.n < 0n ? -this.n : this.n) - bitLen(this.d) - 1;
  }
  /** √q to within an ulp for any size of q ≥ 0 (scaled by an even power of two first, so nothing under- or overflows). */
  sqrtToNumber(): number {
    if (this.n < 0n) return NaN;
    if (this.n === 0n) return 0;
    const k = Math.floor((bitLen(this.n) - bitLen(this.d)) / 2);
    const scaled = k >= 0 ? Q.of(this.n, this.d << BigInt(2 * k)) : Q.of(this.n << BigInt(-2 * k), this.d);
    return scale2(Math.sqrt(scaled.toNumber()), k);
  }
  /** Lossless text: "3", "-5/4". Parsed back by `parseExact`. */
  toString() { return this.d === 1n ? String(this.n) : `${this.n}/${this.d}`; }
  /** Exact square root when this is the square of a rational, else null. */
  sqrtExact(): Q | null {
    if (this.n < 0n) return null;
    const a = isqrt(this.n), b = isqrt(this.d);
    return a * a === this.n && b * b === this.d ? Q.of(a, b) : null;
  }
  /** this^k for an integer k (negative k inverts; 0^negative throws). */
  pow(k: number): Q {
    if (!Number.isInteger(k)) throw new RangeError('integer exponent only');
    if (k < 0) {
      if (this.n === 0n) throw new RangeError('division by zero');
      return Q.of(this.d ** BigInt(-k), this.n ** BigInt(-k));
    }
    return Q.of(this.n ** BigInt(k), this.d ** BigInt(k));
  }
  inv(): Q {
    return Q.ONE.div(this);
  }
  /** Largest integer ≤ this. */
  floor(): bigint {
    const q = this.n / this.d; // truncates toward 0
    return this.n < 0n && q * this.d !== this.n ? q - 1n : q;
  }
  static min(a: Q, b: Q) {
    return a.cmp(b) <= 0 ? a : b;
  }
  static max(a: Q, b: Q) {
    return a.cmp(b) >= 0 ? a : b;
  }
}

function bitLen(x: bigint) { return x === 0n ? 0 : x.toString(2).length; }

/** Integer square root (floor) of a non-negative BigInt. */
/** x·2ᵏ in steps that stay inside the double range (each step exact while the result is a normal number). */
export function scale2(x: number, k: number): number {
  while (k > 1000) {
    x *= 2 ** 1000;
    k -= 1000;
  }
  while (k < -1000) {
    x *= 2 ** -1000;
    k += 1000;
  }
  return x * 2 ** k;
}

export function isqrt(n: bigint): bigint {
  if (n < 0n) throw new RangeError('negative');
  if (n < 2n) return n;
  // Newton from above (2^⌈bits/2⌉ ≥ √n): the iterates decrease to ⌊√n⌋ exactly, for any size
  let x = 1n << BigInt(Math.ceil(bitLen(n) / 2));
  for (;;) {
    const y = (x + n / x) >> 1n;
    if (y >= x) return x;
    x = y;
  }
}

/** n = s² · t with t having no square factor below 10⁴ (t is square-free when n < 10⁸; larger n may keep a big
 *  square factor in t, which is still exact, only less simplified). */
export function splitSquare(n: bigint): { s: bigint; t: bigint } {
  if (n < 0n) throw new RangeError('negative');
  let s = 1n, t = n;
  for (let p = 2n; p < 10000n && p * p <= t; p += p === 2n ? 1n : 2n) {
    while (t % (p * p) === 0n) { t /= p * p; s *= p; }
  }
  const r = isqrt(t);
  if (r > 1n && r * r === t) { s *= r; t = 1n; }
  return { s, t };
}

const MAX_DIGITS = 6;
/**
 * Parse what a reader types: an integer or decimal with at most six decimal places ("−1.25", ".5", "3."), or a
 * fraction of integers ("3/4", "-7/2"). The real minus sign is accepted. Anything else is null — never a silent guess.
 * `maxDigits` allows longer decimals (D22 decides whether a typed axis enters the region from the digits typed:
 * 0.99999999999999999 is inside, though its double is 1).
 */
export function parseExact(text: string, maxDigits = MAX_DIGITS): Q | null {
  const s = text.trim().replace(/[−–]/g, '-').replace(/\s+/g, '');
  if (!s) return null;
  let m = s.match(/^([+-]?)(\d+)\/(\d+)$/);
  if (m) {
    if (m[2].length > 9 || m[3].length > 9) return null;
    const d = BigInt(m[3]);
    if (d === 0n) return null;
    return Q.of((m[1] === '-' ? -1n : 1n) * BigInt(m[2]), d);
  }
  m = s.match(/^([+-]?)(\d*)(?:\.(\d*))?$/);
  if (!m || (!m[2] && !m[3])) return null;
  const int = m[2] || '0', frac = m[3] ?? '';
  if (frac.length > maxDigits || int.length > 9) return null;
  return Q.of((m[1] === '-' ? -1n : 1n) * BigInt(int + frac), 10n ** BigInt(frac.length));
}

/**
 * The decimal a double prints as (`String(x)`: 0.1 → 1/10, −2.75 → −11/4), exactly — what the reader sees — when it
 * has at most six decimal places; null otherwise (0.333…, 1e-7, Infinity). For values a drag, a wheel, a preset or a
 * link produced: the reader means the decimal on the screen, not the double's binary value (`Q.fromDouble`).
 */
export function shownDecimal(x: number): Q | null {
  return Number.isFinite(x) ? parseExact(String(x)) : null;
}

/**
 * Any finite decimal, exactly, with an exponent if written: "0.0000001" is 1/10⁷, "1e-13" is 1/10¹³, "-2.5E3" is
 * −2500 — what a reader means by a decimal, beyond `parseExact`'s six places. The real minus sign is accepted;
 * fractions, π and roots are not (null). At most 40 digits and an exponent within ±400.
 */
export function parseDecimal(text: string): Q | null {
  const m = text.trim().replace(/[−–]/g, '-').match(/^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d{1,3}))?$/);
  if (!m || (!m[2] && !m[3])) return null;
  const digits = (m[2] || '0') + (m[3] ?? '');
  if (digits.length > 40) return null;
  const e = Number(m[4] ?? 0) - (m[3] ?? '').length;
  if (Math.abs(e) > 400) return null;
  const n = (m[1] === '-' ? -1n : 1n) * BigInt(digits);
  return e >= 0 ? Q.of(n * 10n ** BigInt(e)) : Q.of(n, 10n ** BigInt(-e));
}

/** `Q.toString()` read back exactly, whatever its size ("3", "-5/4", "1/100000000000"): a share link's lossless form
 *  of a value the page derived (an output A·v of exact entries). Null for anything else; at most 400 characters. */
export function parseQText(text: string): Q | null {
  const m = text.trim().replace(/[−–]/g, '-').match(/^(-?\d{1,400})(?:\/(\d{1,400}))?$/);
  if (!m) return null;
  const d = m[2] === undefined ? 1n : BigInt(m[2]);
  return d === 0n ? null : Q.of(BigInt(m[1]), d);
}

/** A double rounded to a multiple of `step` (an exact rational step such as 1/10 or 1/20). */
export function roundTo(v: number, step: Q): Q {
  const k = Math.round(v / step.toNumber());
  if (!Number.isFinite(k)) throw new RangeError('not a finite number'); // callers check their range first
  return Q.of(BigInt(k)).mul(step);
}

/** A tidy step for dragging at `raw` world units per few pixels: 1, 2 or 5 × 10ᵏ, exact, never below 0.01. */
export function niceStepQ(raw: number): Q {
  if (!(raw > 0) || !Number.isFinite(raw)) return Q.of(1, 10);
  const e = Math.floor(Math.log10(raw));
  const m = raw / 10 ** e;
  const mult = m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10;
  const step = e >= 0 ? Q.of(BigInt(mult) * 10n ** BigInt(e)) : Q.of(mult, 10n ** BigInt(-e));
  return step.cmp(Q.of(1, 100)) < 0 ? Q.of(1, 100) : step;
}

/** Terminating decimal digits of q if it has at most `max` decimal places, else null ("1.25", "-0.5", "3"). */
export function exactDecimal(q: Q, max = 6): string | null {
  let d = q.d, twos = 0, fives = 0;
  while (d % 2n === 0n) { d /= 2n; twos++; }
  while (d % 5n === 0n) { d /= 5n; fives++; }
  if (d !== 1n) return null;
  const places = Math.max(twos, fives);
  if (places > max) return null;
  const scaled = (q.n * 10n ** BigInt(places)) / q.d; // exact
  const neg = scaled < 0n;
  const digits = babs(scaled).toString().padStart(places + 1, '0');
  const intPart = digits.slice(0, digits.length - places);
  const fracPart = places ? digits.slice(-places) : '';
  return `${neg ? '-' : ''}${intPart}${fracPart ? `.${fracPart}` : ''}`;
}

/**
 * The best rational approximation of x with denominator ≤ maxDen (continued fractions and the best semiconvergent).
 * `exact` is true only when the result equals x as a double. Use it to label a computed double with a fraction the
 * page can show ("≈ 1/6") — never to pretend a float is exact.
 */
export function approximate(x: number, maxDen = 1000): { q: Q; exact: boolean } | null {
  if (!Number.isFinite(x)) return null;
  const sign = x < 0 ? -1n : 1n;
  let v = Math.abs(x);
  // convergents h/k
  let h0 = 0n, h1 = 1n, k0 = 1n, k1 = 0n;
  const max = BigInt(Math.max(1, Math.floor(maxDen)));
  for (let i = 0; i < 64; i++) {
    const a = BigInt(Math.floor(v));
    const h2 = a * h1 + h0, k2 = a * k1 + k0;
    if (k2 > max) {
      // best semiconvergent with denominator ≤ max
      const t = (max - k0) / k1;
      const hs = t * h1 + h0, ks = t * k1 + k0;
      const cand = [Q.of(sign * h1, k1), Q.of(sign * hs, ks)];
      const best = Math.abs(cand[0].toNumber() - x) <= Math.abs(cand[1].toNumber() - x) ? cand[0] : cand[1];
      return { q: best, exact: best.toNumber() === x };
    }
    h0 = h1; h1 = h2; k0 = k1; k1 = k2;
    const frac = v - Math.floor(v);
    if (frac < 1e-15 || !Number.isFinite(1 / frac)) break;
    v = 1 / frac;
  }
  const q = Q.of(sign * h1, k1);
  return { q, exact: q.toNumber() === x };
}

/** KaTeX for a rational: "3", "-\tfrac{5}{4}" style is left to the page; this gives "\frac{5}{4}" with the sign outside. */
export function texOf(q: Q): string {
  if (q.isInteger()) return q.n.toString();
  const neg = q.n < 0n;
  return `${neg ? '-' : ''}\\frac{${(neg ? -q.n : q.n).toString()}}{${q.d.toString()}}`;
}
