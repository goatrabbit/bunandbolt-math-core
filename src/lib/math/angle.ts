// Angles as the trigonometry pages hold them (lifted from C01's page code, 2026-10-07, so C02 and the pages after it
// share one copy; design/pages/C01-unit-circle.md "数学", design/pages/C02-sine-and-cosine.md 改动 14). Pure: no DOM.
//
// θ has two states (C01's Codex review). Exact: θ is a rational multiple of π, held exactly as q = θ/π (a Q) —
// whatever was typed as a π fraction or in degrees, a degree step from an exact angle, the special angle a drag
// catches on, a value key, a link's π form. Only the special angles (multiples of π/6 and π/4) show closed forms
// (√3/2) and "does not exist", both from the declared table below, never from a float or a tolerance; other exact
// angles show decimals and stay exact. Numeric: decimal radians, a wheel, a free drag, a timed run — always decimals
// with ≈; a reciprocal too large for a double says so with its order of magnitude, never "does not exist".
import { Q, exactDecimal } from './rational';
import { piRational, readTyped } from '../ui/typed';

const PI = Math.PI;

const mod = (a: number, n: number) => ((a % n) + n) % n;

/** θ = k·π/12 is one of the special angles: a multiple of π/6 (k even) or of π/4 (k a multiple of 3). */
export const isSpecialK = (k: number) => Number.isInteger(k) && (k % 2 === 0 || k % 3 === 0);
/** The 16 special angles of one turn, in twelfths of π. */
export const SPECIAL_K = Array.from({ length: 24 }, (_, i) => i).filter(isSpecialK);
export const thetaOf = (k: number) => (k * PI) / 12;

// ---------- the angle ----------

/** θ as a page holds it: q = θ/π exactly when θ is exact, else null; theta (radians) always. */
export interface Angle { theta: number; q: Q | null }
/** (n·π)/d, the way a typed "nπ/d" is read (src/lib/ui/typed.ts), so the two agree. */
const thetaOfQ = (q: Q) => (Number(q.n) * PI) / Number(q.d);
export const exactQ = (q: Q): Angle => ({ theta: thetaOfQ(q), q });
/** An exact angle in twelfths of π (the special angles: k = 8 is 2π/3). */
export const exactAngle = (k: number) => exactQ(Q.of(k, 12));
/** A numeric θ; exactly 0 (or −0) is the exact angle 0. */
export const numericAngle = (theta: number): Angle => (theta === 0 ? exactQ(Q.ZERO) : { theta, q: null });
/** k when θ is exactly one of the special angles (any turn), else null. */
export function specialK(a: Angle): number | null {
  if (!a.q || 12n % a.q.d !== 0n) return null;
  const k = Number(a.q.n * (12n / a.q.d));
  return isSpecialK(k) ? k : null;
}
export const sameAngle = (a: Angle, b: Angle) => (a.q && b.q ? a.q.eq(b.q) : !a.q && !b.q && a.theta === b.theta);
/** The exact angle a + q·π (θ + π/2 is `plusQ(a, Q.of(1, 2))`); numeric when a is. */
export const plusQ = (a: Angle, q: Q): Angle => (a.q ? exactQ(a.q.add(q)) : numericAngle(a.theta + thetaOfQ(q)));

/** A range of angles given as multiples of π (C01: −20 … 20; C02: 0 … 2), checked exactly for an exact angle. */
export interface AngleRange { lo: Q; hi: Q }
export const piRange = (lo: number, hi: number): AngleRange => ({ lo: Q.of(lo), hi: Q.of(hi) });
/** Within the range; an exact angle is checked exactly (3600° is 20π, kept as ten turns, not reduced) and only while
 *  its numerator and denominator stay within a double's integers. */
export function angleIn(a: Angle, r: AngleRange): boolean {
  if (a.q) return a.q.cmp(r.lo) >= 0 && a.q.cmp(r.hi) <= 0 && a.q.d <= 2n ** 53n && (a.q.n < 0n ? -a.q.n : a.q.n) <= 2n ** 53n;
  return Number.isFinite(a.theta) && a.theta >= thetaOfQ(r.lo) && a.theta <= thetaOfQ(r.hi);
}

// ---------- the declared table ----------

/** The absolute values that occur, written once (0, 1/2, √2/2 … 2√3/3, √2). */
export type Abs = '0' | '1/2' | '√2/2' | '√3/2' | '1' | '√3' | '√3/3' | '2' | '2√3/3' | '√2';
export const ABS_VALUE: Record<Abs, number> = {
  '0': 0, '1/2': 0.5, '√2/2': Math.SQRT2 / 2, '√3/2': Math.sqrt(3) / 2, '1': 1, '√3': Math.sqrt(3), '√3/3': Math.sqrt(3) / 3,
  '2': 2, '2√3/3': (2 * Math.sqrt(3)) / 3, '√2': Math.SQRT2,
};

export type Fn = 'cos' | 'sin' | 'tan' | 'cot' | 'sec' | 'csc';
export const FNS: Fn[] = ['cos', 'sin', 'tan', 'cot', 'sec', 'csc'];

/** A value of one function at θ: undefined (only at an exact special angle), exact (a closed form), a number, or a
 *  number too large for a double (its sign and order of magnitude). */
export type Val =
  | { kind: 'undef' }
  | { kind: 'exact'; sign: -1 | 0 | 1; abs: Abs; value: number }
  | { kind: 'num'; value: number }
  | { kind: 'big'; sign: -1 | 1; exp: number };

/** By reference angle (0, π/6, π/4, π/3, π/2 in twelfths): |f| for each function, null where it does not exist. */
export const BY_REF: Record<0 | 2 | 3 | 4 | 6, Record<Fn, Abs | null>> = {
  0: { cos: '1', sin: '0', tan: '0', cot: null, sec: '1', csc: null },
  2: { cos: '√3/2', sin: '1/2', tan: '√3/3', cot: '√3', sec: '2√3/3', csc: '2' },
  3: { cos: '√2/2', sin: '√2/2', tan: '1', cot: '1', sec: '√2', csc: '√2' },
  4: { cos: '1/2', sin: '√3/2', tan: '√3', cot: '√3/3', sec: '2', csc: '2√3/3' },
  6: { cos: '0', sin: '1', tan: null, cot: '0', sec: null, csc: '1' },
};

/** The reference angle of k (in twelfths, within one quadrant) and the signs of cos and sin there: the absolute
 *  values come from the reference triangle, the signs from the quadrant. */
export function reference(k: number): { ref: 0 | 2 | 3 | 4 | 6; cos: -1 | 0 | 1; sin: -1 | 0 | 1 } {
  const r = mod(k, 24);
  const ref = (r <= 6 ? r : r <= 12 ? 12 - r : r <= 18 ? r - 12 : 24 - r) as 0 | 2 | 3 | 4 | 6;
  const cos = r === 6 || r === 18 ? 0 : r > 6 && r < 18 ? -1 : 1;
  const sin = r === 0 || r === 12 ? 0 : r > 12 ? -1 : 1;
  return { ref, cos, sin };
}

/** The six functions at a special angle, from the table: closed forms with their signs; undefined where the function
 *  does not exist (tan, sec at π/2 + nπ; cot, csc at nπ). */
export function exactValues(k: number): Record<Fn, Val> {
  const { ref, cos, sin } = reference(k);
  const sgn: Record<Fn, number> = { cos, sin, tan: cos * sin, cot: cos * sin, sec: cos, csc: sin };
  const out = {} as Record<Fn, Val>;
  for (const f of FNS) {
    const abs = BY_REF[ref][f];
    if (abs === null) { out[f] = { kind: 'undef' }; continue; }
    const sign = (abs === '0' ? 0 : sgn[f]) as -1 | 0 | 1;
    out[f] = { kind: 'exact', sign, abs, value: abs === '0' ? 0 : sign * ABS_VALUE[abs] };
  }
  return out;
}

/** a / b as a value: a double when it is one, else its sign and order of magnitude (1/sin(1e−309)). */
function ratio(a: number, b: number): Val {
  const v = a / b;
  if (Number.isFinite(v)) return { kind: 'num', value: v };
  return { kind: 'big', sign: Math.sign(a) * Math.sign(b) < 0 ? -1 : 1, exp: b === 0 ? Infinity : Math.floor(Math.log10(Math.abs(a)) - Math.log10(Math.abs(b))) };
}
/** The six functions as decimals (a numeric θ, or an exact angle that is not a special one). */
export function numericValues(theta: number): Record<Fn, Val> {
  const c = Math.cos(theta), s = Math.sin(theta);
  return { cos: { kind: 'num', value: c }, sin: { kind: 'num', value: s }, tan: ratio(s, c), cot: ratio(c, s), sec: ratio(1, c), csc: ratio(1, s) };
}
export function valuesAt(a: Angle): Record<Fn, Val> {
  const k = specialK(a);
  return k !== null ? exactValues(k) : numericValues(a.theta);
}
/** A value as a double for drawing (a big one is very large but finite; undefined is NaN). */
export const drawValue = (v: Val) => (v.kind === 'undef' ? NaN : v.kind === 'big' ? v.sign * 1e300 : v.value);

// ---------- writing numbers ----------

const minus = (s: string) => s.replace(/-/g, '−');
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
export const sup = (n: number) => `${n < 0 ? '⁻' : ''}${String(Math.abs(n)).split('').map((d) => SUP[+d]).join('')}`;
/** The one rule for decimals (readouts and the picture alike): four places below 10⁴, then five significant digits
 *  times a power of ten; a true minus sign, never "−0.0000". */
export function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1e4) {
    const [m, e] = v.toExponential(4).split('e');
    return `${minus(m)}×10${sup(Number(e))}`;
  }
  const s = v.toFixed(4);
  return /^-0\.0+$/.test(s) ? s.slice(1) : minus(s);
}
/** A value as the readouts write it, without its relation: "−√3/2" exactly, "0.5403" as a decimal; null where it does
 *  not exist; "10³⁰⁹" for one too large for a double (the page says "too large" with the exponent). */
export function valText(v: Val): string | null {
  if (v.kind === 'undef') return null;
  if (v.kind === 'exact') return `${v.sign < 0 ? '−' : ''}${v.abs}`;
  if (v.kind === 'big') return `${v.sign < 0 ? '−' : ''}10${sup(v.exp)}`;
  return fmtNum(v.value);
}
export const relOf = (v: Val) => (v.kind === 'exact' ? '=' : '≈');

/** q·π written with π: 0, π, −π/2, 2π/3, 61π/360, 20π (`ascii`: "2pi/3", for links). */
export function piText(q: Q, ascii = false): string {
  if (q.isZero()) return '0';
  const pi = ascii ? 'pi' : 'π';
  const a = q.n < 0n ? -q.n : q.n;
  return `${q.n < 0n ? (ascii ? '-' : '−') : ''}${a === 1n ? '' : a}${pi}${q.d === 1n ? '' : `/${q.d}`}`;
}
/** θ in radians as a readout writes it, with its relation: the π form when exact; a numeric θ as typed when that is
 *  exactly it (1, 2.0944), else to four places with ≈. */
export function radText(a: Angle): { rel: '=' | '≈'; text: string } {
  if (a.q) return { rel: '=', text: piText(a.q) };
  return Number(a.theta.toFixed(4)) === a.theta ? { rel: '=', text: minus(String(a.theta)) } : { rel: '≈', text: fmtNum(a.theta) };
}
const fmtDeg = (v: number) => { const s = v.toFixed(4); return /^-0\.0+$/.test(s) ? s.slice(1) : minus(s); };
/** θ in degrees: exactly when it terminates within four places (120°, 30.5°), else ≈ to four places (θ = 1 is
 *  ≈ 57.2958°; 0.5236 is ≈ 30.0001°, not 30°). */
export function degText(a: Angle): { rel: '=' | '≈'; text: string } {
  if (a.q) {
    const d = a.q.mul(Q.of(180));
    const exact = exactDecimal(d, 4);
    if (exact !== null) return { rel: '=', text: `${minus(exact)}°` };
    return { rel: '≈', text: `${fmtDeg(d.toNumber())}°` };
  }
  return { rel: '≈', text: `${fmtDeg((a.theta * 180) / PI)}°` };
}

// ---------- typing θ ----------

export type AngleRead = { ok: true; angle: Angle } | { ok: false; reason: 'nan' | 'range' };

/** What a typed angle means: a π fraction or any number of degrees is exact (120° = 2π/3, 30.5°, 15°); a decimal in
 *  radians is numeric (0.5236 stays ≈ 30.0001°). Outside `range` it is refused. */
export function readAngle(text: string, range: AngleRange): AngleRead {
  const r = readTyped(text);
  if (!r.ok) return { ok: false, reason: 'nan' };
  const q = piRational(text);
  const angle = q ? exactQ(q) : numericAngle(r.value);
  return angleIn(angle, range) ? { ok: true, angle } : { ok: false, reason: 'range' };
}

/** The text an angle field shows while edited, which reads back as the same state: an exact angle as its canonical π
 *  form, a numeric one as the shortest decimal that reads back as the same double — never a guessed π (editText would
 *  turn 0.5235987755982988 into π/6 and Enter would make it exact). */
export const editAngle = (a: Angle) => (a.q ? piText(a.q) : String(a.theta));

// ---------- moving θ ----------

/** The special angle nearest θ, in twelfths of π (any turn). */
export function nearestSpecialK(theta: number): number {
  const base = Math.round(theta / (PI / 12));
  let best = base, bd = Infinity;
  for (let d = -2; d <= 2; d++) {
    const k = base + d;
    if (!isSpecialK(k)) continue;
    const e = Math.abs(theta - thetaOf(k));
    if (e < bd) { bd = e; best = k; }
  }
  return best;
}

/** A drag's catch (the knob's and the direction dial's feel): a special angle catches θ within `grab` radians of the
 *  hand's travel and, once caught, holds it until the hand is `hold` away. Null: not caught. */
export function snapK(theta: number, caught: number | null, grab: number, hold: number): number | null {
  if (caught !== null && Math.abs(theta - thetaOf(caught)) < hold) return caught;
  const k = nearestSpecialK(theta);
  return Math.abs(theta - thetaOf(k)) < grab ? k : null;
}

/** The next special angle after θ (dir 1, counter-clockwise) or before it (−1), in twelfths of π. Decided exactly for
 *  an exact angle (31° → 45°), from the double for a numeric one. */
export function stepSpecial(a: Angle, dir: 1 | -1): number {
  let k: number;
  if (a.q) {
    const x = a.q.mul(Q.of(12));
    k = dir > 0 ? Number(x.floor()) + 1 : -Number(x.neg().floor()) - 1;
  } else {
    k = dir > 0 ? Math.floor(a.theta / (PI / 12) + 1e-9) + 1 : Math.ceil(a.theta / (PI / 12) - 1e-9) - 1;
  }
  while (!isSpecialK(k)) k += dir;
  return k;
}

/** Degree steps (an arrow key: one degree, a tenth with Shift), in tenths of a degree. From an exact angle the step
 *  is exact (back to 120° is 2π/3 again); from a numeric one it stays numeric. */
export function stepDegrees(a: Angle, tenths: number): Angle {
  if (a.q) return exactQ(a.q.add(Q.of(tenths, 1800)));
  return numericAngle(a.theta + (tenths * PI) / 1800);
}

/** The angle of a special angle clicked on a circle (one turn, `r` twelfths): the coterminal angle nearest the current
 *  θ (at 390°, the 45° tick is 405°), so a point slides the shorter way and θ stays continuous; a tie (half a turn
 *  away) goes counter-clockwise. */
export function tickTarget(a: Angle, r: number): number {
  const cur = a.theta / (PI / 12);
  let k = r + 24 * Math.round((cur - r) / 24);
  if (k - cur > 12) k -= 24;
  if (cur - k > 12) k += 24;
  return k;
}
