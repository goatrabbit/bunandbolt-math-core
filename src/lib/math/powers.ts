// Exact positive reals of the form e^a · ∏ gᵢ^rᵢ: integers gᵢ > 1, rational exponents a and rᵢ (2026-10-07, A12
// logarithm). Every number a reader types as a decimal, a fraction, a root k√r/d or e is one of these, and so is every
// product, quotient and rational power of them — so a page can decide exactly whether log_b x is rational, and which
// rational, instead of trusting Math.log(1000) / Math.log(10) = 2.9999999999999996.
//
// The decision rests on one fact: 1, ln 2, ln 3, ln 5, … are linearly independent over ℚ (unique prime factorisation,
// and e^q is transcendental for rational q ≠ 0 while ∏pᵢ^rᵢ is algebraic). Hence log_b x = ln x / ln b is rational
// exactly when x's exponent vector (its e exponent and its prime exponents) is a rational multiple λ of b's, and then
// log_b x = λ. No number is ever factored into primes: two numbers are written over a common base of pairwise coprime
// integers, refined by gcds (on such a base the exponent vectors are as good as prime ones for proportionality).
import { Q, splitSquare } from './rational';

const gcd = (a: bigint, b: bigint): bigint => {
  while (b) [a, b] = [b, a % b];
  return a;
};

/** Pairwise coprime integers > 1 such that each input (> 1) is a product of powers of them (factor refinement:
 *  a pair with g = gcd > 1 is replaced by a/g, b/g and g until no pair shares a factor). */
export function coprimeBase(list: bigint[]): bigint[] {
  let S = [...new Set(list.filter((x) => x > 1n))];
  for (;;) {
    let split = false;
    outer: for (let i = 0; i < S.length; i++) {
      for (let j = i + 1; j < S.length; j++) {
        const g = gcd(S[i], S[j]);
        if (g === 1n) continue;
        const next = S.filter((_, k) => k !== i && k !== j);
        next.push(S[i] / g, S[j] / g, g);
        S = [...new Set(next.filter((x) => x > 1n))];
        split = true;
        break outer;
      }
    }
    if (!split) return S.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  }
}

/** ⌊n^(1/j)⌋ for n ≥ 0 (Newton from above, exact for any size). */
function iroot(n: bigint, j: number): bigint {
  if (n < 2n) return n;
  const J = BigInt(j);
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / j));
  for (;;) {
    const y = ((J - 1n) * x + n / x ** (J - 1n)) / J;
    if (y >= x) return x;
    x = y;
  }
}
const POWERS = new Map<bigint, [bigint, number]>();
/** n = h^j with j as large as it goes (n > 1): 1002001 is 1001², 64 is 2⁶. A generator is kept free of perfect powers,
 *  so a value like √1.002001 = 1001/1000 is seen to be rational (Codex implementation review 6). */
function perfectPower(n: bigint): [bigint, number] {
  const hit = POWERS.get(n);
  if (hit) return hit;
  let out: [bigint, number] = [n, 1];
  for (let j = n.toString(2).length; j >= 2; j--) {
    const h = iroot(n, j);
    if (h > 1n && h ** BigInt(j) === n) { out = [h, j]; break; }
  }
  if (POWERS.size > 5000) POWERS.clear();
  POWERS.set(n, out);
  return out;
}
const lcm = (a: bigint, b: bigint) => (a / gcd(a, b)) * b;

/** How many times s divides n (s > 1). */
function valuation(n: bigint, s: bigint): number {
  let k = 0;
  while (n % s === 0n) { n /= s; k++; }
  return k;
}

/** Two numbers' exponents over one coprime base: [e, then one entry per base element]. */
function vectors(a: Pow, b: Pow): [Q[], Q[], bigint[]] {
  const base = coprimeBase([...a.f.keys(), ...b.f.keys()]);
  const over = (p: Pow) => {
    const v: Q[] = [p.e, ...base.map(() => Q.ZERO)];
    for (const [g, r] of p.f) base.forEach((s, i) => { const k = valuation(g, s); if (k) v[i + 1] = v[i + 1].add(r.mul(Q.of(k))); });
    return v;
  };
  return [over(a), over(b), base];
}

/** A positive real e^e · ∏ g^f(g), exact. Immutable; the generators of one value are pairwise coprime. */
export class Pow {
  private constructor(readonly e: Q, readonly f: ReadonlyMap<bigint, Q>) {}

  static readonly ONE = new Pow(Q.ZERO, new Map());
  static readonly E = new Pow(Q.ONE, new Map());

  /** A positive rational. */
  static ofQ(q: Q): Pow {
    if (q.sign() <= 0) throw new RangeError('a power product is positive');
    const f = new Map<bigint, Q>();
    // n and d are coprime: already a coprime base (each taken as a power of its smallest root)
    if (q.n > 1n) { const [h, j] = perfectPower(q.n); f.set(h, Q.of(j)); }
    if (q.d > 1n) { const [h, j] = perfectPower(q.d); f.set(h, Q.of(-j)); }
    return new Pow(Q.ZERO, f);
  }
  static of(n: number | bigint, d: number | bigint = 1): Pow {
    return Pow.ofQ(Q.of(n, d));
  }

  /** Over a coprime base, without zero exponents, each generator a power of its smallest root (the roots of coprime
   *  numbers are coprime). */
  private static build(e: Q, base: bigint[], v: Q[]): Pow {
    const f = new Map<bigint, Q>();
    base.forEach((s, i) => {
      if (v[i].isZero()) return;
      const [h, j] = perfectPower(s);
      f.set(h, v[i].mul(Q.of(j)));
    });
    return new Pow(e, f);
  }

  mul(o: Pow): Pow {
    const [a, b, base] = vectors(this, o);
    return Pow.build(a[0].add(b[0]), base, base.map((_, i) => a[i + 1].add(b[i + 1])));
  }
  div(o: Pow): Pow {
    return this.mul(o.pow(Q.of(-1)));
  }
  /** this^k for a rational k. */
  pow(k: Q): Pow {
    if (k.isZero()) return Pow.ONE;
    const f = new Map<bigint, Q>();
    for (const [g, r] of this.f) f.set(g, r.mul(k));
    return new Pow(this.e.mul(k), f);
  }
  /** this^k for an integer k. */
  powInt(k: number): Pow {
    return this.pow(Q.of(k));
  }

  isOne(): boolean {
    return this.e.isZero() && this.f.size === 0;
  }
  eq(o: Pow): boolean {
    const [a, b] = vectors(this, o);
    return a.every((x, i) => x.eq(b[i]));
  }
  /** The exact rational when every exponent is an integer and there is no e, else null. */
  toQ(): Q | null {
    if (!this.e.isZero()) return null;
    let q = Q.ONE;
    for (const [g, r] of this.f) {
      if (!r.isInteger()) return null;
      q = q.mul(Q.of(g).pow(Number(r.n)));
    }
    return q;
  }

  /** ln of this, as a double. Without e, some power thisᴸ (L the exponents' common denominator, up to 64) is an exact
   *  rational: ln = ln(thisᴸ) / L, through log1p near 1, so b = 1.001 and √(1 + 10⁻²⁰) keep their digits (Codex
   *  implementation review 4). Otherwise e + Σ r·ln g. */
  ln(): number {
    if (this.e.isZero()) {
      let L = 1n;
      for (const r of this.f.values()) { L = lcm(L, r.d); if (L > 64n) break; }
      const q = L <= 64n ? (L === 1n ? this : this.pow(Q.of(L))).toQ() : null;
      if (q) {
        const near = q.sub(Q.ONE), l = Number(L);
        if (near.abs().cmp(Q.of(1, 2)) < 0) return Math.log1p(near.toNumber()) / l;
        const v = q.toNumber();
        return (v > 0 && Number.isFinite(v) ? Math.log(v) : lnBig(q.n) - lnBig(q.d)) / l;
      }
    }
    let s = this.e.toNumber();
    for (const [g, r] of this.f) s += r.toNumber() * lnBig(g);
    return s;
  }
  /** The nearest double: the integer parts of the exponents exactly (a Q), the fractional parts and e through exp. */
  toNumber(): number {
    let q = Q.ONE, rest = this.e.toNumber();
    for (const [g, r] of this.f) {
      const whole = r.floor();
      q = q.mul(Q.of(g).pow(Number(whole)));
      const frac = r.sub(Q.of(whole));
      if (!frac.isZero()) rest += frac.toNumber() * lnBig(g);
    }
    const qv = q.toNumber();
    const v = rest === 0 ? qv : qv * Math.exp(rest);
    if (v > 0 && Number.isFinite(v)) return v;
    // a part overflows or underflows on its own though the product need not (2²⁰⁰⁰·e⁻¹³⁸⁶ ≈ 1.342): add logarithms
    return Math.exp(lnBig(q.n) - lnBig(q.d) + rest);
  }

  /**
   * log_base(this) when it is rational, else null (see the header). Throws when the base is 1 (no logarithm).
   * log_b 1 = 0 for every base.
   */
  logIn(base: Pow): Q | null {
    const [x, b] = vectors(this, base);
    const c = b.findIndex((v) => !v.isZero());
    if (c < 0) throw new RangeError('the base of a logarithm cannot be 1');
    const lambda = x[c].div(b[c]);
    return x.every((v, i) => v.eq(b[i].mul(lambda))) ? lambda : null;
  }

  /** this = q · √r · e^k with q rational, r a square-free-ish integer ≥ 1 (`splitSquare`) and k rational, when every
   *  prime-side exponent is a multiple of ½; null otherwise (a cube root, say). For writing the value out. */
  surd(): { q: Q; r: bigint; e: Q } | null {
    let q = Q.ONE, r = 1n;
    for (const [g, x] of this.f) {
      const twice = x.mul(Q.of(2));
      if (!twice.isInteger()) return null;
      const t = Number(twice.n);
      const whole = Math.floor(t / 2);
      q = q.mul(Q.of(g).pow(whole));
      if (t - 2 * whole === 1) r *= g;
    }
    const { s, t } = splitSquare(r);
    return { q: q.mul(Q.of(s)), r: t, e: this.e };
  }
}

/** ln of a positive BigInt, beyond the double range too. */
function lnBig(n: bigint): number {
  const v = Number(n);
  if (Number.isFinite(v)) return Math.log(v);
  const bits = n.toString(2).length, shift = bits - 1000;
  return Math.log(Number(n >> BigInt(shift))) + shift * Math.LN2;
}
