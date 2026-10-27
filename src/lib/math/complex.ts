// Complex numbers as a page holds them (2026-10-07, A15 complex plane): the double value always, and alongside it
// whatever is known exactly — the rectangular parts when both are rational, |c|² when it is rational, and arg c as a
// rational fraction of a turn. Each operation keeps what it can and drops the rest, so a page prints "=" only for what
// is exact and "≈" for the rest, and decides questions such as "is wⁿ = 1?" without trusting a float.
//
// The rules rest on three facts:
// - Products: rect·rect is rational; |ab|² = |a|²|b|²; arg ab ≡ arg a + arg b (mod 2π) when a, b ≠ 0.
// - For rational a, b (not both 0), arg(a + bi) / 2π is rational only on the axes and the diagonals (a = 0, b = 0 or
//   |a| = |b|): a rational tangent at a rational multiple of π is 0 or ±1 (from Niven's theorem; e.g. Olmsted 1945).
//   So arg(2 + i) has no exact number of degrees, and `turn` is null for it.
// - At a rational turn, cos² is rational exactly at the multiples of 30° and 45° (cos² θ = (1 + cos 2θ)/2, and by
//   Niven cos 2θ is rational only when it is 0, ±1/2 or ±1): there cos² ∈ {0, 1/4, 1/2, 3/4, 1}. With |c|² rational,
//   the squares of the parts are then rational, and the parts are rational when those are squares (e^{iπ/3} =
//   1/2 + (√3/2)i has a rational real part only). At any other rational turn with |c| ≠ 0 the parts are irrational, so
//   `rect` is null there by the mathematics, not for want of trying.
import { Q } from './rational';
import { Surd } from './surd';

export interface Cx {
  /** the value as doubles */
  re: number;
  im: number;
  /** both parts exactly, when both are rational */
  rect: readonly [Q, Q] | null;
  /** |c|² exactly, when rational */
  mod2: Q | null;
  /** arg c / 2π exactly, in (−1/2, 1/2], when it is rational and known; null for 0 */
  turn: Q | null;
}

const HALF = Q.of(1, 2);
const TAU = 2 * Math.PI;

/** A turn reduced to the principal value, (−1/2, 1/2]. */
export function principalTurn(t: Q): Q {
  // t − round-half-down(t): the fractional part shifted into (−1/2, 1/2]
  let r = t.sub(Q.of(t.floor()));
  if (r.cmp(HALF) > 0) r = r.sub(Q.ONE);
  return r;
}

/** The exact turn of a rational point, where there is one: the axes and the diagonals (see the header); else null. */
export function turnOfRect(a: Q, b: Q): Q | null {
  const sa = a.sign(), sb = b.sign();
  if (sa === 0 && sb === 0) return null;
  if (sb === 0) return sa > 0 ? Q.ZERO : HALF;
  if (sa === 0) return Q.of(sb, 4);
  if (!a.abs().eq(b.abs())) return null;
  return Q.of(sb * (sa > 0 ? 1 : 3), 8);
}

/** cos² at a turn that is a multiple of 1/12 (30°) or 1/8 (45°), exactly; null at any other turn. */
function cos2(t: Q): Q | null {
  const k12 = t.mul(Q.of(12)), k8 = t.mul(Q.of(8));
  if (k12.isInteger()) {
    const k = Number(((k12.n % 12n) + 12n) % 12n); // multiples of 30°, one turn
    return [Q.ONE, Q.of(3, 4), Q.of(1, 4), Q.ZERO, Q.of(1, 4), Q.of(3, 4), Q.ONE, Q.of(3, 4), Q.of(1, 4), Q.ZERO, Q.of(1, 4), Q.of(3, 4)][k];
  }
  if (k8.isInteger()) return HALF; // the odd multiples of 45° (the even ones are multiples of 90°, above)
  return null;
}

/** Exact rational parts from exact |c|² and turn, where the special angles allow it (see the header); else null. */
export function rectOfPolar(mod2: Q, turn: Q): [Q, Q] | null {
  if (mod2.isZero()) return [Q.ZERO, Q.ZERO];
  const c2 = cos2(turn);
  if (!c2) return null;
  const re = mod2.mul(c2).sqrtExact(), im = mod2.mul(Q.ONE.sub(c2)).sqrtExact();
  if (!re || !im) return null;
  // signs from the quadrant of the turn (principal value: cos ≥ 0 on [−1/4, 1/4], sin ≥ 0 on [0, 1/2])
  const t = principalTurn(turn);
  const sc = t.abs().cmp(Q.of(1, 4)) > 0 ? -1 : 1;
  const ss = t.sign() < 0 ? -1 : 1;
  return [sc < 0 ? re.neg() : re, ss < 0 ? im.neg() : im];
}

/** cos and sin of a turn as doubles, exact on the axes (cos π/2 is 0, not 6e−17). */
function cisTurn(t: number): [number, number] {
  const q4 = t * 4;
  if (Number.isInteger(q4)) {
    const k = ((q4 % 4) + 4) % 4;
    return [[1, 0], [0, 1], [-1, 0], [0, -1]][k] as [number, number];
  }
  return [Math.cos(TAU * t), Math.sin(TAU * t)];
}

/** Fill in what follows from what is known: rect gives mod2 and (on axes and diagonals) turn; mod2 and a special turn
 *  give rect. The value comes from the most exact source. */
function complete(c: Cx): Cx {
  let { rect, mod2, turn } = c;
  if (rect) {
    mod2 = rect[0].mul(rect[0]).add(rect[1].mul(rect[1]));
    turn = turnOfRect(rect[0], rect[1]);
    return { re: rect[0].toNumber(), im: rect[1].toNumber(), rect, mod2, turn };
  }
  if (mod2 && mod2.isZero()) return { re: 0, im: 0, rect: [Q.ZERO, Q.ZERO], mod2, turn: null };
  if (mod2 && turn) {
    const r = rectOfPolar(mod2, turn);
    if (r) return complete({ ...c, rect: r });
  }
  if (turn) {
    // the direction exactly, the length as known
    const m = mod2 ? mod2.sqrtToNumber() : Math.hypot(c.re, c.im);
    const [co, si] = cisTurn(turn.toNumber());
    return { re: m * co, im: m * si, rect: null, mod2, turn };
  }
  if (mod2) {
    // the length exactly: rescale the double value onto it
    // the direction first, then the length: (re / h)·m, never (re·m) / h, which underflows for a tiny m (Codex 6)
    const h = Math.hypot(c.re, c.im), m = mod2.sqrtToNumber();
    return h > 0 ? { re: (c.re / h) * m, im: (c.im / h) * m, rect: null, mod2, turn: null } : { ...c, rect: null, turn: null };
  }
  return { re: c.re, im: c.im, rect: null, mod2: null, turn: null };
}

/** Exactly a + bi. */
export const cxRect = (a: Q, b: Q): Cx => complete({ re: 0, im: 0, rect: [a, b], mod2: null, turn: null });
/** A numeric value (a drag, a wheel): nothing claimed. */
export const cxNum = (re: number, im: number): Cx => ({ re: re + 0, im: im + 0, rect: null, mod2: null, turn: null });

/** A modulus as a page may know it: its double, and its exact square when that is rational. */
export interface Real { v: number; sq: Q | null }
/** A direction as a page may know it: arg / 2π as a double, and exactly when it is rational. */
export interface Turn { v: number; q: Q | null }
export const realQ = (q: Q): Real => ({ v: q.toNumber(), sq: q.mul(q) });
export const turnQ = (q: Q): Turn => ({ v: q.toNumber(), q });

/** r·e^{2πit}. A zero modulus gives exactly 0 when the modulus is exact. A negative r is not a modulus: refused. */
export function cxPolar(r: Real, t: Turn): Cx {
  if (!(r.v >= 0)) throw new RangeError('a modulus is not negative');
  // exactly 0 only when the exact square is 0: a non-zero square whose double underflowed is not 0 (Codex 6)
  if (r.sq?.isZero()) return cxRect(Q.ZERO, Q.ZERO);
  if (r.v === 0 && !r.sq) return cxNum(0, 0);
  const [co, si] = cisTurn(t.v);
  return complete({ re: r.v * co, im: r.v * si, rect: null, mod2: r.sq, turn: t.q ? principalTurn(t.q) : null });
}

export const ZERO: Cx = cxRect(Q.ZERO, Q.ZERO);
export const ONE: Cx = cxRect(Q.ONE, Q.ZERO);
export const I: Cx = cxRect(Q.ZERO, Q.ONE);

/** Exactly zero, or a double value of 0 + 0i. */
export const isZero = (c: Cx) => (c.rect ? c.rect[0].isZero() && c.rect[1].isZero() : c.mod2 ? c.mod2.isZero() : c.re === 0 && c.im === 0);
/** |c| as a double (from the exact square when known). */
export const modulus = (c: Cx) => (c.mod2 ? c.mod2.sqrtToNumber() : Math.hypot(c.re, c.im));
/** arg c in radians, principal value (−π, π]; NaN for 0. */
export function argRad(c: Cx): number {
  if (isZero(c)) return NaN;
  if (c.turn) return TAU * c.turn.toNumber();
  const a = Math.atan2(c.im, c.re);
  return a === -Math.PI ? Math.PI : a;
}
/** Both exactly the same number, or both numeric with the same doubles. */
export function sameCx(a: Cx, b: Cx): boolean {
  if (a.rect && b.rect) return a.rect[0].eq(b.rect[0]) && a.rect[1].eq(b.rect[1]);
  return a.re === b.re && a.im === b.im && !!a.rect === !!b.rect && eqQ(a.mod2, b.mod2) && eqQ(a.turn, b.turn);
}
const eqQ = (a: Q | null, b: Q | null) => (a && b ? a.eq(b) : a === b);
/** Provably the same number: equal rational parts, or equal exact |·|² and turn, or both exactly 0. Two doubles that
 *  happen to agree prove nothing (Codex implementation review 7). */
export function provablyEqual(a: Cx, b: Cx): boolean {
  const z = (c: Cx) => !!(c.rect || c.mod2) && isZero(c);
  if (z(a) || z(b)) return z(a) && z(b);
  if (a.rect && b.rect) return a.rect[0].eq(b.rect[0]) && a.rect[1].eq(b.rect[1]);
  return !!(a.mod2 && b.mod2 && a.turn && b.turn) && a.mod2!.eq(b.mod2!) && a.turn!.eq(b.turn!);
}

export function cxMul(a: Cx, b: Cx): Cx {
  if ((isZero(a) && (a.rect || a.mod2)) || (isZero(b) && (b.rect || b.mod2))) return ZERO; // 0·w = 0 exactly
  if (a.rect && b.rect) {
    const [p, q] = a.rect, [r, s] = b.rect;
    return cxRect(p.mul(r).sub(q.mul(s)), p.mul(s).add(q.mul(r)));
  }
  return complete({
    re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re, rect: null,
    mod2: a.mod2 && b.mod2 ? a.mod2.mul(b.mod2) : null,
    turn: a.turn && b.turn ? principalTurn(a.turn.add(b.turn)) : null,
  });
}

/** a + b: rational parts add exactly; parts known as ±√q add exactly when both are rational or they cancel
 *  (e^{iπ/3} + e^{−iπ/3} = 1, Codex design review 6); two numbers with the same exact turn keep it. 0 + b is b. */
export function cxAdd(a: Cx, b: Cx): Cx {
  if (a.rect && b.rect) return cxRect(a.rect[0].add(b.rect[0]), a.rect[1].add(b.rect[1]));
  if (isZero(a) && (a.rect || a.mod2)) return b;
  if (isZero(b) && (b.rect || b.mod2)) return a;
  const pa = exactParts(a), pb = exactParts(b);
  if (pa && pb) {
    const signed = (x: SignedRoot) => { const r = x.sq.sqrtExact(); return r && (x.sign < 0 ? r.neg() : r); };
    const sum = (x: SignedRoot, y: SignedRoot): Q | null => {
      const rx = signed(x), ry = signed(y);
      if (rx && ry) return rx.add(ry);
      return x.sq.eq(y.sq) && x.sign === -y.sign ? Q.ZERO : null;
    };
    const re = sum(pa[0], pb[0]), im = sum(pa[1], pb[1]);
    if (re && im) return cxRect(re, im);
  }
  const turn = a.turn && b.turn && a.turn.eq(b.turn) ? a.turn : null;
  return complete({ re: a.re + b.re, im: a.im + b.im, rect: null, mod2: null, turn });
}

/** cⁿ for an integer n ≥ 0; c⁰ = 1 (also for c = 0: an orbit starts where it is). */
export function cxPow(c: Cx, n: number): Cx {
  if (!Number.isInteger(n) || n < 0) throw new RangeError('n is a whole number');
  // the repeated product keeps every exact part: |c|²ⁿ, n·turn (reduced), the rational parts
  let out = ONE;
  for (let k = 0; k < n; k++) out = cxMul(out, c);
  return out;
}

/** √q = s·√t with s rational and t a square-free integer (t = 1 when √q is rational); q ≥ 0. The decomposition is
 *  surd.ts's `Surd.sqrt`. */
export function surd(q: Q): { s: Q; t: bigint } {
  const r = Surd.sqrt(q);
  if (!r) throw new RangeError('negative');
  return r.isRational() ? { s: r.rational()!, t: 1n } : { s: r.b, t: r.t };
}

/** arg c in degrees, principal value (−180°, 180°]: exactly (a rational) when the turn is known; null for 0. */
export function argDeg(c: Cx): { exact: Q | null; value: number } | null {
  if (isZero(c)) return null;
  if (c.turn) { const d = c.turn.mul(Q.of(360)); return { exact: d, value: d.toNumber() }; }
  return { exact: null, value: (argRad(c) * 180) / Math.PI };
}

/** A real number known as ±√sq exactly (sq rational ≥ 0). */
export interface SignedRoot { sign: -1 | 0 | 1; sq: Q }
/** The parts as ±√q, exactly, when known: from rational parts, or from |c|² and a turn at a multiple of 30° or 45°
 *  (e^{iπ/3} = 1/2 + (√3/2)i); null otherwise (and at any other turn the parts are not of this form). */
export function exactParts(c: Cx): [SignedRoot, SignedRoot] | null {
  if (c.rect) return [{ sign: c.rect[0].sign(), sq: c.rect[0].mul(c.rect[0]) }, { sign: c.rect[1].sign(), sq: c.rect[1].mul(c.rect[1]) }];
  if (!c.mod2 || !c.turn) return null;
  const c2 = cos2(c.turn);
  if (!c2) return null;
  const re = c.mod2.mul(c2), im = c.mod2.mul(Q.ONE.sub(c2));
  const t = principalTurn(c.turn);
  const sc = re.isZero() ? 0 : t.abs().cmp(Q.of(1, 4)) > 0 ? -1 : 1;
  const ss = im.isZero() ? 0 : t.sign() < 0 ? -1 : 1;
  return [{ sign: sc, sq: re }, { sign: ss, sq: im }];
}
