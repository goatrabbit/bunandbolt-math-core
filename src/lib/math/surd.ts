// Exact numbers of the form a + b·√t (a, b rational, t a positive integer with no square factor below 10⁴), for pages
// whose exact readouts pass through one square root: a solid of revolution's slice area π(R² − r²) with R = 2 + √(1 − x²)
// (D22), a circle's chord, a quadratic's roots. Closed under +, −, × when the roots agree; anything that would need two
// different roots (√2 + √3) gives null — the page then shows a rounded value, never a guessed exact one.
//
// `Surd.sqrt` takes the root of a rational: √(n/d) = √(n·d)/d, and n·d = s²·t (`splitSquare`), so √q = (s/d)·√t, which
// is rational exactly when t = 1. Equal t means equal roots (a larger square factor left in t only loses some
// exactness, never correctness).
//
// 2026-10-08 (B04 Pythagorean theorem): the exact sign (is a signed piece negative?), division by the conjugate, the
// exact cosines of the turns where cos is a + b√t (`cosOfTurn`), and a short written form for readouts (`surdForm`).
import { Q, exactDecimal, splitSquare } from './rational';

export class Surd {
  private constructor(readonly a: Q, readonly b: Q, readonly t: bigint) {}
  static of(a: Q | number | bigint): Surd {
    return new Surd(a instanceof Q ? a : Q.of(a), Q.ZERO, 1n);
  }
  /** √q for q ≥ 0; null for a negative q. */
  static sqrt(q: Q): Surd | null {
    if (q.sign() < 0) return null;
    if (q.isZero()) return Surd.of(0);
    const { s, t } = splitSquare(q.n * q.d);
    const c = Q.of(s, q.d);
    return t === 1n ? Surd.of(c) : new Surd(Q.ZERO, c, t);
  }
  /** the root this number carries (1 when rational) */
  get root(): bigint { return this.b.isZero() ? 1n : this.t; }
  isRational(): boolean { return this.b.isZero(); }
  /** the rational value, or null when there is a root part */
  rational(): Q | null { return this.b.isZero() ? this.a : null; }
  private static join(x: Surd, y: Surd): bigint | null {
    if (x.b.isZero()) return y.root;
    if (y.b.isZero()) return x.root;
    return x.t === y.t ? x.t : null;
  }
  add(o: Surd): Surd | null {
    const t = Surd.join(this, o);
    if (t === null) return null;
    const b = this.b.add(o.b);
    return new Surd(this.a.add(o.a), b, b.isZero() ? 1n : t);
  }
  neg(): Surd { return new Surd(this.a.neg(), this.b.neg(), this.t); }
  sub(o: Surd): Surd | null { return this.add(o.neg()); }
  mul(o: Surd): Surd | null {
    const t = Surd.join(this, o);
    if (t === null) return null;
    // (a + b√t)(c + d√t) = ac + bd·t + (ad + bc)√t
    const a = this.a.mul(o.a).add(this.b.mul(o.b).mul(Q.of(t)));
    const b = this.a.mul(o.b).add(this.b.mul(o.a));
    return new Surd(a, b, b.isZero() ? 1n : t);
  }
  scale(q: Q): Surd { return new Surd(this.a.mul(q), this.b.mul(q), this.b.mul(q).isZero() ? 1n : this.t); }
  sq(): Surd { return this.mul(this)!; }
  /** the nearest double (a + b·√t, each part rounded once) */
  toNumber(): number { return this.a.toNumber() + (this.b.isZero() ? 0 : this.b.toNumber() * Math.sqrt(Number(this.t))); }
  isZero(): boolean { return this.a.isZero() && this.b.isZero(); }
  /** The exact sign of a + b√t: the parts' common sign, or the larger of a² and b²·t when they differ (2026-10-08, B04:
   *  is the piece 9 − 6√3 negative?). */
  sign(): -1 | 0 | 1 {
    const sa = this.a.sign(), sb = this.b.sign();
    if (sb === 0) return sa;
    if (sa === 0 || sa === sb) return sb;
    const c = this.a.mul(this.a).cmp(this.b.mul(this.b).mul(Q.of(this.t)));
    return c === 0 ? 0 : c > 0 ? sa : sb;
  }
  /** Equal as numbers (a root part that is zero has no root, so the parts are canonical). */
  eq(o: Surd): boolean { return this.a.eq(o.a) && this.b.eq(o.b) && (this.b.isZero() || this.t === o.t); }
  /** this / o, by the conjugate: (a + b√t)/(c + d√t) = (a + b√t)(c − d√t)/(c² − d²t); null for o = 0 or two roots. */
  div(o: Surd): Surd | null {
    if (o.isZero()) return null;
    const conj = new Surd(o.a, o.b.neg(), o.t);
    const top = this.mul(conj);
    if (!top) return null;
    const den = o.mul(conj)!.rational()!; // c² − d²t: rational, and not 0 (√t is irrational)
    return top.scale(den.inv());
  }
}

/** cos(2π·turn) exactly where it is a + b√t: at the multiples of 30° and 45° (0, ±1/2, ±1, ±√3/2, ±√2/2) and of 36°
 *  (±(1 ± √5)/4); null at any other turn (cos 15° = (√6 + √2)/4 needs two roots, cos 18° a nested one). A rational
 *  cosine at a rational turn is one of 0, ±1/2, ±1 (Niven), and cos is quadratic over ℚ exactly at the turns with
 *  denominator 5, 8, 10 or 12 among the rest. */
export function cosOfTurn(turn: Q): Surd | null {
  const half = Q.of(1, 2), quarter = Q.of(1, 4);
  // reduce to [0, 1/2] (cos is even and 1-periodic in turns)
  let t = turn.sub(Q.of(turn.floor()));
  if (t.cmp(half) > 0) t = Q.ONE.sub(t);
  const deg = t.mul(Q.of(360));
  if (!deg.isInteger()) return null;
  const k = Number(deg.n);
  const r2 = Surd.sqrt(Q.of(2))!.scale(half), r3 = Surd.sqrt(Q.of(3))!.scale(half), r5 = Surd.sqrt(Q.of(5))!.scale(quarter);
  const table: Record<number, Surd> = {
    0: Surd.of(1), 30: r3, 45: r2, 60: Surd.of(half), 90: Surd.of(0), 120: Surd.of(half.neg()), 135: r2.neg(), 150: r3.neg(), 180: Surd.of(-1),
    36: Surd.of(quarter).add(r5)!, 72: r5.sub(Surd.of(quarter))!, 108: Surd.of(quarter).sub(r5)!, 144: Surd.of(quarter).add(r5)!.neg(),
  };
  return table[k] ?? null;
}

/** A short exact form of a + b√t for a readout, or null when one would be long: each rational part a terminating
 *  decimal with at most `places` decimals, or a fraction with a denominator up to 12; the coefficient of √t written
 *  as "6√3", "√13", "−√2/2". `tex`: KaTeX (\sqrt{3}, \tfrac{}{}); otherwise text with a real minus sign. */
export function surdForm(s: Surd, places = 4, tex = false): string | null {
  const part = (q: Q): string | null => {
    const d = exactDecimal(q, places);
    if (d !== null) return d;
    if (q.d > 12n) return null;
    const n = q.n < 0n ? -q.n : q.n;
    return tex ? `\\tfrac{${n}}{${q.d}}` : `${n}/${q.d}`;
  };
  const minus = tex ? '-' : '−';
  const root = tex ? `\\sqrt{${s.t}}` : `√${s.t}`;
  if (s.b.isZero()) {
    const p = part(s.a.abs());
    return p === null ? null : `${s.a.sign() < 0 ? minus : ''}${p}`;
  }
  // b√t: "√13", "6√3", "√3/2" (the denominator after the root), "1.5√2"
  const b = s.b.abs();
  let rootPart: string | null;
  if (b.isInteger()) rootPart = `${b.n === 1n ? '' : b.n}${root}`;
  else if (b.d <= 12n) rootPart = tex ? `\\tfrac{${b.n === 1n ? '' : b.n}${root}}{${b.d}}` : `${b.n === 1n ? '' : b.n}${root}/${b.d}`;
  else if (exactDecimal(b, places) !== null) rootPart = `${exactDecimal(b, places)}${root}`;
  else rootPart = null;
  if (rootPart === null) return null;
  if (s.a.isZero()) return `${s.b.sign() < 0 ? minus : ''}${rootPart}`;
  const a = part(s.a.abs());
  if (a === null) return null;
  // a positive part first: "25√3/4 − 9" rather than "−9 + 25√3/4"
  if (s.a.sign() < 0 && s.b.sign() > 0) return `${rootPart} ${minus} ${a}`;
  return `${s.a.sign() < 0 ? minus : ''}${a} ${s.b.sign() < 0 ? minus : '+'} ${rootPart}`;
}

/** a·b for possibly-missing exact values: null when either is missing or they cannot be combined. */
export const sMul = (x: Surd | null, y: Surd | null) => (x && y ? x.mul(y) : null);
export const sAdd = (x: Surd | null, y: Surd | null) => (x && y ? x.add(y) : null);
export const sSub = (x: Surd | null, y: Surd | null) => (x && y ? x.sub(y) : null);
