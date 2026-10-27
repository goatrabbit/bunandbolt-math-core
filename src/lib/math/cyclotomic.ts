// Exact zero tests for sums of cosines and sines of rational angles (2026-10-08, B01 rigid transformation): is
// c + Σ (aⱼ cos βⱼ + bⱼ sin βⱼ) exactly 0, for rational aⱼ, bⱼ, c and angles βⱼ that are a rational number of degrees?
// A page that keeps its angles exact (B01's mirrors, to the thousandth of a degree) decides "are these mirrors the
// same line", "is this a reflection or a glide", "is this point already on its target" with it, never with a float
// tolerance — 89.9999° is not 90°.
//
// Why it works: write βⱼ = 360°·kⱼ/N with one N for all (a multiple of 4), and ζ = e^{2πi/N}. Then
// cos βⱼ = (ζ^k + ζ^{−k})/2 and sin βⱼ = (ζ^{k+3N/4} + ζ^{N/4−k})/2 (i = ζ^{N/4}), so the sum is P(ζ) for a polynomial P
// with rational coefficients and degree < N. ζ is a root of the cyclotomic polynomial Φ_N, which is irreducible over
// Q, so P(ζ) = 0 exactly when Φ_N divides P. With m the product of N's distinct primes, Φ_N(x) = Φ_m(x^{N/m}), and the
// monomials x^r·y^j (r < N/m, j < φ(m), y = x^{N/m}) are a basis of Q[x]/Φ_N: so P(ζ) = 0 exactly when, for every
// residue r mod N/m, the polynomial in y formed by P's exponents ≡ r is divisible by Φ_m. For angles in thousandths of
// a degree m divides 30 and Φ₃₀ has degree 8: a test costs a few dozen BigInt operations.
import { Q } from './rational';

/** One term a·cos β + b·sin β, with β a rational number of degrees. */
export interface TrigTerm { deg: Q; cos: Q; sin: Q }

const babs = (x: bigint) => (x < 0n ? -x : x);
function gcd(a: bigint, b: bigint): bigint {
  a = babs(a);
  b = babs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}
const lcm = (a: bigint, b: bigint) => (a / gcd(a, b)) * b;
const mod = (a: bigint, n: bigint) => ((a % n) + n) % n;

/** Above this the squarefree part of N makes Φ_m large; angles with such denominators are refused. */
export const MAX_RADICAL = 5000;

/** The distinct primes of n (n ≥ 1), searched only as far as MAX_RADICAL: a larger prime factor is refused at once
 *  (the radical would be too large anyway), so a huge denominator never costs a long trial division. */
function primes(n: bigint): bigint[] {
  const out: bigint[] = [];
  const limit = BigInt(MAX_RADICAL);
  for (let p = 2n; p * p <= n && p <= limit; p++) {
    if (n % p) continue;
    out.push(p);
    while (n % p === 0n) n /= p;
  }
  if (n > limit) throw new RangeError('angles too fine to decide exactly (a prime factor beyond MAX_RADICAL)');
  if (n > 1n) out.push(n);
  return out;
}

const PHI = new Map<number, bigint[]>();
/** Φ_m's integer coefficients (index = degree), m squarefree: Π_{d | m} (x^d − 1)^{μ(m/d)}. */
function cyclotomic(m: number, ps: bigint[]): bigint[] {
  const hit = PHI.get(m);
  if (hit) return hit;
  // the divisors d of m with μ(m/d): every subset of m's primes is m/d
  const up: number[] = [], down: number[] = [];
  for (let s = 0; s < 1 << ps.length; s++) {
    let q = 1, k = 0;
    ps.forEach((p, i) => { if (s & (1 << i)) { q *= Number(p); k++; } });
    (k % 2 ? down : up).push(m / q);
  }
  let poly: bigint[] = [1n];
  for (const d of up) { // × (x^d − 1)
    const next = new Array<bigint>(poly.length + d).fill(0n);
    poly.forEach((c, i) => { next[i + d] += c; next[i] -= c; });
    poly = next;
  }
  for (const d of down) { // ÷ (x^d − 1), exact: q[i] = p[i + d] + q[i + d]
    const n = poly.length - 1, q = new Array<bigint>(n - d + 1).fill(0n);
    for (let i = n - d; i >= 0; i--) q[i] = poly[i + d] + (i + d <= n - d ? q[i + d] : 0n);
    poly = q;
  }
  PHI.set(m, poly);
  return poly;
}

/** Is P(ζ_N) = 0, for P given by its integer coefficients at exponents (any integers: they are taken mod N, and
 *  coefficients at the same exponent mod N are added)? N a multiple of 4. */
export function rootOfUnitySumIsZero(N: bigint, coef0: Map<bigint, bigint>): boolean {
  const coef = new Map<bigint, bigint>();
  for (const [e, c] of coef0) { const k = mod(e, N); coef.set(k, (coef.get(k) ?? 0n) + c); }
  const ps = primes(N);
  const m = ps.reduce((a, p) => a * p, 1n);
  if (m > BigInt(MAX_RADICAL)) throw new RangeError(`angles too fine to decide exactly (radical ${m})`);
  const s = N / m;
  const phi = cyclotomic(Number(m), ps);
  const deg = phi.length - 1;
  // the polynomial in y for each residue r mod N/m
  const groups = new Map<bigint, Map<number, bigint>>();
  for (const [e, c] of coef) {
    if (c === 0n) continue;
    const r = e % s, j = Number(e / s);
    let g = groups.get(r);
    if (!g) groups.set(r, (g = new Map()));
    g.set(j, (g.get(j) ?? 0n) + c);
  }
  for (const g of groups.values()) {
    let top = -1;
    for (const [j, c] of g) if (c !== 0n && j > top) top = j;
    if (top < 0) continue;
    const r = new Array<bigint>(top + 1).fill(0n);
    for (const [j, c] of g) r[j] += c;
    // remainder by the monic Φ_m
    for (let i = top; i >= deg; i--) {
      const c = r[i];
      if (c === 0n) continue;
      for (let t = 0; t <= deg; t++) r[i - deg + t] -= c * phi[t];
    }
    for (let i = 0; i < Math.min(deg, r.length); i++) if (r[i] !== 0n) return false;
  }
  return true;
}

/** Is c + Σ (aⱼ cos βⱼ + bⱼ sin βⱼ) exactly zero? Angles in degrees, any rational number of them (their denominators,
 *  times 360, must not bring primes whose product exceeds MAX_RADICAL; thousandths of a degree bring 2, 3 and 5). */
export function trigSumIsZero(c: Q, terms: TrigTerm[]): boolean {
  const live = terms.filter((t) => !t.cos.isZero() || !t.sin.isZero());
  if (!live.length) return c.isZero();
  // one N for every angle: β/360 = k/N, N a multiple of 4 so that i = ζ^{N/4}
  const turns = live.map((t) => t.deg.div(Q.of(360)));
  let N = 4n;
  for (const r of turns) N = lcm(N, r.d);
  // a common denominator for the coefficients (each cos and sin brings a ½)
  let D = 2n * c.d;
  for (const t of live) D = lcm(D, 2n * lcm(t.cos.d, t.sin.d));
  const coef = new Map<bigint, bigint>();
  const add = (e: bigint, q: Q) => {
    if (q.isZero()) return;
    const k = mod(e, N);
    coef.set(k, (coef.get(k) ?? 0n) + (q.n * D) / q.d);
  };
  add(0n, c);
  const half = Q.of(1, 2);
  live.forEach((t, i) => {
    const k = (turns[i].n * N) / turns[i].d;
    add(k, t.cos.mul(half));
    add(-k, t.cos.mul(half));
    add(k + (3n * N) / 4n, t.sin.mul(half));
    add(N / 4n - k, t.sin.mul(half));
  });
  return rootOfUnitySumIsZero(N, coef);
}

/** Is a·cos β + b·sin β + c exactly zero (β in degrees)? */
export const trigIsZero = (c: Q, a: Q, b: Q, deg: Q) => trigSumIsZero(c, [{ deg, cos: a, sin: b }]);
