// What a value field accepts (design spec §8.0.1): one reading of typed numbers for every page, so a form that works
// on one page works on all of them. Pure; no DOM.
//
//   1   −2.5   .5   3.   1e-6   2.5E3     decimals and scientific notation (true minus and en dash are minus signs)
//   1,5   31,536,000                       in each number, one comma is a decimal point; groups of three are thousands
//   1/3   −7/2                             fractions
//   π   pi   2π   −π/2   3π/4   0.5π       multiples and fractions of π
//   √3   −√2   2√3   √3/2                  square roots, with a coefficient and a denominator
//   e   −e                                 Euler's number
//   120°   120deg   −30°   22.5°           degrees, read as radians: 120° is exactly the value of 2π/3 (2026-10-07, C01)
//
// Full-width characters (１２．５, －) are read as their ASCII forms. Anything else is refused, never guessed: "0x10",
// "Infinity", "1e999", superscripts and vulgar fractions ("2²", "1½"), and a non-zero literal too small for a double
// ("1e-999", which would silently become 0 — a special point on many graphs) all fail.

import { Q } from '../math/rational';

export type TypedFail = 'empty' | 'nan' | 'underflow';
export type Typed = { ok: true; value: number } | { ok: false; reason: TypedFail };

const DEC = String.raw`(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?`;
const RE_DEC = new RegExp(`^${DEC}$`);
const RE_FRAC = /^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/;
const RE_PI = new RegExp(`^(${DEC})?[*·×]?π(?:\\/(\\d+(?:\\.\\d+)?))?$`);
const RE_ROOT = new RegExp(`^(${DEC})?[*·×]?√(\\d+(?:\\.\\d+)?)(?:\\/(\\d+(?:\\.\\d+)?))?$`);
const RE_DEG = new RegExp(`^(${DEC})(?:°|deg)$`);

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
/** Degrees as a fraction of π, num / den in lowest terms ("120" → 2 / 3, "22.5" → 1 / 8), so an angle typed in degrees
 *  is the same number as its π form: (num·π) / den is how a typed "2π/3" is computed. Null for what has no such small
 *  fraction (an exponent, too many digits): then (deg·π) / 180. */
function degFraction(dec: string): [number, number] | null {
  const m = dec.match(/^(\d*)(?:\.(\d*))?$/);
  if (!m) return null;
  const frac = m[2] ?? '';
  const num = Number(`${m[1]}${frac}`), den = 180 * 10 ** frac.length;
  if (!Number.isSafeInteger(num) || !Number.isSafeInteger(den)) return null;
  const g = gcd(num, den) || 1;
  return [num / g, den / g];
}

/** One run of digits, points and commas: thousands groups lose their commas, a single comma between digits is a
 *  decimal point; any other comma makes the run unreadable (null). */
function commas(run: string): string | null {
  if (!run.includes(',')) return run;
  if (/^\d{1,3}(,\d{3})+(\.\d*)?$/.test(run)) return run.replace(/,/g, ''); // 31,536,000
  if (/^\d*,\d+$/.test(run)) return run.replace(',', '.'); // 1,5
  return null;
}

/** Normalise what the keyboard (or an IME) produced: full width, minus signs, spaces, "pi", commas. Null if a comma
 *  cannot be read. */
function normalise(text: string): string | null {
  let s = text
    .replace(/[！-～]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0)) // full width → ASCII
    .toLowerCase()
    .replace(/[\s_　]/g, '')
    .replace(/[−–—]/g, '-')
    .replace(/pi/g, 'π');
  let bad = false;
  s = s.replace(/[\d.,]+/g, (run) => {
    const r = commas(run);
    if (r === null) bad = true;
    return r ?? run;
  });
  return bad ? null : s;
}

/** True when the digits before any exponent are not all zero (so a result of 0 means the literal underflowed). */
const nonZeroDigits = (s: string) => /[1-9]/.test(s.split('e')[0]);

export function readTyped(text: string): Typed {
  const raw = normalise(text);
  if (raw === null) return { ok: false, reason: 'nan' };
  if (raw === '') return { ok: false, reason: 'empty' };
  const sign = raw.startsWith('-') ? -1 : 1;
  const s = raw.replace(/^[+-]/, '');
  let v: number;
  /** the literal parts that must not all vanish: a non-zero literal giving 0 underflowed (1e-999, 1e-999π, 1e-999√3) */
  let lits: string[] = [];
  let m: RegExpMatchArray | null;
  if (RE_DEC.test(s)) {
    v = Number(s);
    lits = [s];
  } else if ((m = s.match(RE_FRAC))) {
    const den = Number(m[2]);
    if (den === 0) return { ok: false, reason: 'nan' };
    v = Number(m[1]) / den;
    lits = [m[1]];
  } else if ((m = s.match(RE_PI))) {
    const den = m[2] === undefined ? 1 : Number(m[2]);
    if (den === 0) return { ok: false, reason: 'nan' };
    v = ((m[1] === undefined ? 1 : Number(m[1])) * Math.PI) / den;
    lits = [m[1] ?? '1'];
  } else if ((m = s.match(RE_ROOT))) {
    const den = m[3] === undefined ? 1 : Number(m[3]);
    if (den === 0) return { ok: false, reason: 'nan' };
    v = ((m[1] === undefined ? 1 : Number(m[1])) * Math.sqrt(Number(m[2]))) / den;
    lits = [m[1] ?? '1', m[2]];
  } else if ((m = s.match(RE_DEG))) {
    const f = degFraction(m[1]);
    v = f ? (f[0] * Math.PI) / f[1] : (Number(m[1]) * Math.PI) / 180;
    lits = [m[1]];
  } else if (s === 'e') {
    v = Math.E;
  } else {
    return { ok: false, reason: 'nan' };
  }
  if (!Number.isFinite(v)) return { ok: false, reason: 'nan' };
  if (v === 0 && lits.length && lits.every(nonZeroDigits)) return { ok: false, reason: 'underflow' };
  return { ok: true, value: sign * v + 0 }; // + 0 turns −0 into 0
}

/**
 * The same text as `readTyped`, read exactly, for a quantity a page decides exact questions about (D01's a, L, ε, δ:
 * "is |L − 4| < ε?" must not hang on 4.1 − 0.1 being 3.9999999999999996 in binary). Decimals with any number of
 * places, scientific notation and fractions become the rational they name; π, √ and e are refused as 'irrational'
 * (no exact decision can be made about them); everything `readTyped` refuses is refused for the same reason.
 */
export function readTypedQ(text: string): { ok: true; q: Q; value: number } | { ok: false; reason: TypedFail | 'irrational' } {
  const r = readTyped(text);
  if (!r.ok) return r;
  const raw = normalise(text)!;
  const neg = raw.startsWith('-');
  const s = raw.replace(/^[+-]/, '');
  let q: Q | null;
  let m: RegExpMatchArray | null;
  if (RE_DEC.test(s)) q = decimalQ(s);
  else if ((m = s.match(RE_FRAC))) { const n = decimalQ(m[1]), d = decimalQ(m[2]); q = n && d && !d.isZero() ? n.div(d) : null; }
  else return { ok: false, reason: 'irrational' };
  if (!q) return { ok: false, reason: 'nan' }; // more digits or a larger exponent than an exact reading takes
  q = neg ? q.neg() : q;
  return { ok: true, q, value: q.toNumber() };
}

/** The number typed, or NaN for anything that is not one (blank, junk, overflow, underflow). */
export function parseTyped(text: string): number {
  const r = readTyped(text);
  return r.ok ? r.value : NaN;
}

/** A typed multiple of π as its coefficient ("−π/2" → −0.5, "2π" → 2), exactly; null when the text is not written
 *  with π or in degrees ("30°" → 1/6). For quantities kept in units of π, so π/6 is stored as 1/6 and not as (π/6)/π. */
export function piMultiple(text: string): number | null {
  const raw = normalise(text);
  if (raw === null) return null;
  const sign = raw.startsWith('-') ? -1 : 1;
  const body = raw.replace(/^[+-]/, '');
  const d = body.match(RE_DEG);
  if (d) {
    const v = (sign * Number(d[1])) / 180; // one division: as exact as the π form's (30/180 is 1/6)
    return Number.isFinite(v) && !(v === 0 && nonZeroDigits(d[1])) ? v + 0 : null;
  }
  const m = body.match(RE_PI);
  if (!m) return null;
  const k = m[1] === undefined ? 1 : Number(m[1]);
  const den = m[2] === undefined ? 1 : Number(m[2]);
  const v = (sign * k) / den;
  return Number.isFinite(v) && !(v === 0 && m[1] !== undefined && nonZeroDigits(m[1])) ? v + 0 : null;
}

/** A decimal literal as typed ("2", "0.5", "22.5", "1e-3"), exactly; null when it would need more than 30 digits. */
function decimalQ(lit: string): Q | null {
  const m = lit.match(/^(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$/);
  if (!m) return null;
  const digits = `${m[1]}${m[2] ?? ''}`.replace(/^0+(?=\d)/, '') || '0';
  const e = Number(m[3] ?? 0) - (m[2] ?? '').length;
  if (digits.length > 30 || Math.abs(e) > 30) return null;
  return e >= 0 ? Q.of(BigInt(digits) * 10n ** BigInt(e)) : Q.of(BigInt(digits), 10n ** BigInt(-e));
}

/** A typed multiple of π, or an angle in degrees, as an exact rational multiple of π ("2π/3" → 2/3, "0.5π" → 1/2,
 *  "30.5°" → 61/360, "-3600°" → −20); null for anything else, a decimal in radians included. Decided from the text,
 *  never from a float (2026-10-07, C01: an angle typed this way is exact). */
export function piRational(text: string): Q | null {
  const raw = normalise(text);
  if (raw === null) return null;
  const sign = raw.startsWith('-') ? -1n : 1n;
  const body = raw.replace(/^[+-]/, '');
  const d = body.match(RE_DEG);
  if (d) { const v = decimalQ(d[1]); return v ? Q.of(sign * v.n, v.d * 180n) : null; }
  const m = body.match(RE_PI);
  if (!m) return null;
  const k = m[1] === undefined ? Q.ONE : decimalQ(m[1]);
  const den = m[2] === undefined ? Q.ONE : decimalQ(m[2]);
  if (!k || !den || den.isZero()) return null;
  const v = k.div(den);
  return Q.of(sign * v.n, v.d);
}

/** A typed angle as an exact rational number of degrees, for a field whose unit is the degree (2026-10-08, B06's ∠AOB;
 *  B01 reads its mirrors the same way): a decimal or a fraction is degrees ("40", "200/3"), "40°" is degrees, a π form
 *  is radians turned into degrees ("π/4" is 45). Decided from the text, never from a float. √ and e are refused as
 *  'irrational' (no exact number of degrees), anything `readTyped` refuses as 'nan'. */
export function typedDegreesQ(text: string): { ok: true; deg: Q } | { ok: false; reason: 'nan' | 'irrational' } {
  const pi = piRational(text); // θ/π, for π forms and degree forms
  if (pi) return { ok: true, deg: pi.mul(Q.of(180)) };
  const r = readTypedQ(text);
  if (r.ok) return { ok: true, deg: r.q };
  return { ok: false, reason: r.reason === 'irrational' && readTyped(text).ok ? 'irrational' : 'nan' };
}

/** What was typed, exactly, as coef · √root · eᵏ (k is 0 or 1), for the forms that are algebraic or e: a decimal,
 *  a fraction, a root k√r/d and e, each with its sign in `coef` ("2√3/4" → coef 1/2, root 3; "0.5" → coef 1/2;
 *  "e" → coef 1, k 1). The same number readTyped reads. Null for π forms, degrees, and anything readTyped refuses —
 *  a page that needs exact values (2026-10-07, A12: is log_b x rational?) refuses those rather than guessing. */
export function typedExact(text: string): { coef: Q; root: Q | null; e: 0 | 1 } | null {
  if (!readTyped(text).ok) return null;
  const raw = normalise(text);
  if (raw === null) return null;
  const sign = raw.startsWith('-') ? -1n : 1n;
  const s = raw.replace(/^[+-]/, '');
  const signed = (q: Q) => Q.of(sign * q.n, q.d);
  let m: RegExpMatchArray | null;
  if (RE_DEC.test(s)) { const v = decimalQ(s); return v ? { coef: signed(v), root: null, e: 0 } : null; }
  if ((m = s.match(RE_FRAC))) {
    const a = decimalQ(m[1]), b = decimalQ(m[2]);
    return a && b && !b.isZero() ? { coef: signed(a.div(b)), root: null, e: 0 } : null;
  }
  if ((m = s.match(RE_ROOT))) {
    const k = m[1] === undefined ? Q.ONE : decimalQ(m[1]);
    const r = decimalQ(m[2]);
    const d = m[3] === undefined ? Q.ONE : decimalQ(m[3]);
    return k && r && d && !d.isZero() ? { coef: signed(k.div(d)), root: r, e: 0 } : null;
  }
  if (s === 'e') return { coef: Q.of(sign), root: null, e: 1 };
  return null;
}

/** The number of degrees typed ("120°", "−30deg" → 120, −30), or null when the text is not written in degrees. For a
 *  page that treats a whole number of degrees as exact (C01's special angles). */
export function typedDegrees(text: string): number | null {
  const raw = normalise(text);
  if (raw === null) return null;
  const m = raw.replace(/^[+-]/, '').match(RE_DEG);
  if (!m) return null;
  const v = (raw.startsWith('-') ? -1 : 1) * Number(m[1]);
  return Number.isFinite(v) ? v + 0 : null;
}

/** What a value field shows while it is being edited: text that reads back as exactly `v` (design spec §8.0.1), so
 *  focusing a field and pressing Enter changes nothing. Short forms first — twelve significant digits when they are
 *  exact, a multiple of π or a √ form when `v` is one (π/2 stays π/2, not 1.5707963268) — then the shortest decimal
 *  that reads back as `v`.
 *
 *  WARNING: the π and √ forms are guessed from the float. A numeric value that happens to be the double nearest 2π/3
 *  (dragged there, or typed as 2.0943951023931953) comes back as "2π/3", and a page that keeps exact values apart from
 *  numeric ones (C01's angle: a rational multiple of π, or a plain number) would read that text back as exact — a
 *  focus and Enter would turn a numeric state into an exact one. Such a page writes its edit text from its state:
 *  `editTextQ`. */
export function editText(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const short = String(+v.toPrecision(12));
  if (parseTyped(short) === v) return short;
  const sign = v < 0 ? '-' : '';
  for (let d = 1; d <= 12; d++) {
    for (let k = 1; k <= 24; k++) {
      const t = `${sign}${k === 1 ? '' : k}π${d === 1 ? '' : `/${d}`}`;
      if (parseTyped(t) === v) return t;
    }
  }
  for (const r of [2, 3, 5, 6, 7]) {
    for (let d = 1; d <= 4; d++) {
      for (let k = 1; k <= 4; k++) {
        const t = `${sign}${k === 1 ? '' : k}√${r}${d === 1 ? '' : `/${d}`}`;
        if (parseTyped(t) === v) return t;
      }
    }
  }
  return String(v);
}

/** The edit text for a page whose state knows whether a value is exact: `q` is the exact value when the state holds
 *  one, null when the value is numeric. Never guesses from the float (unlike `editText`):
 *  - exact: written from `q` — with `pi: true` as a multiple of π ("2π/3", "-11π/6", "20π"), read back exactly with
 *    `piRational`, and by `parseTyped` as (n·π)/d — compute the page's double that way, not as q.toNumber()·π, which
 *    can differ in the last place (π/13); zero is "0", which `piRational` does not read as a multiple of π, so a page
 *    treats a field left as it was as no change (C01 does). Without `pi`, as a fraction ("2/3", "-7/2", "3"), read back
 *    exactly with `parseExact`. Numerators and denominators up to 30 digits (what the typed readers take);
 *  - numeric: the shortest decimal that reads back as `v` ("2.0943951023931953", never "2π/3"), so a numeric state stays
 *    numeric through a focus and Enter. */
export function editTextQ(v: number, q: Q | null, o: { pi?: boolean } = {}): string {
  if (!q) return Number.isFinite(v) ? String(v + 0) : String(v);
  if (!o.pi) return q.toString();
  if (q.isZero()) return '0';
  const a = q.n < 0n ? -q.n : q.n;
  return `${q.n < 0n ? '-' : ''}${a === 1n ? '' : a}π${q.d === 1n ? '' : `/${q.d}`}`;
}

// ---------- complex numbers (2026-10-07, A15 complex plane) ----------

/** A real number as typed: its value, exactly when it is rational (`q`), and its exact square when that is rational
 *  (`sq`: also for a root k√r/d, so a typed modulus √2 has |w|² = 2 exactly). */
export interface TypedReal { value: number; q: Q | null; sq: Q | null }
/** An angle as typed: radians, and exactly as a fraction of a turn when written in degrees or with π (`turn`, not
 *  reduced: "400°" is 10/9). A bare number is radians and numeric, as everywhere on the site. */
export interface TypedAngle { value: number; turn: Q | null }
export type TypedComplex =
  | { ok: true; form: 'rect'; re: TypedReal; im: TypedReal }
  | { ok: true; form: 'polar'; r: TypedReal; angle: TypedAngle }
  | { ok: false; reason: TypedFail };

function typedReal(text: string): TypedReal | TypedFail {
  const r = readTyped(text);
  if (!r.ok) return r.reason;
  const x = typedExact(text);
  const q = x && x.e === 0 && !x.root ? x.coef : null;
  const sq = x && x.e === 0 ? (x.root ? x.coef.mul(x.coef).mul(x.root) : x.coef.mul(x.coef)) : null;
  return { value: r.value, q, sq };
}
function typedAngle(text: string): TypedAngle | TypedFail {
  const r = readTyped(text);
  if (!r.ok) return r.reason;
  const pq = piRational(text);
  return { value: r.value, turn: pq ? Q.of(pq.n, pq.d * 2n) : null };
}
const fail = (reason: TypedFail): TypedComplex => ({ ok: false, reason });

/**
 * A complex number as a reader types it, in the site's one reading of numbers (`readTyped` for every part):
 *
 *   2+i   2 - 0.5i   -i   3i   i/2   1/2+√3/2i   1/2+i√3/2   e+πi        rectangular: a ± bi, b before or after i
 *   1.5∠40°   2∠π/3   2<120deg   1∠-1.2                                 polar r∠θ ("<" for ∠; a bare θ is radians)
 *   e^(iπ/3)   2e^(iπ/6)   e^{2πi/5}   3·e^(i90°)                        exponential r·e^(iθ)
 *
 * A negative modulus is refused (−2∠30° would hide a half turn). Anything else is refused, never guessed.
 */
export function typedComplex(text: string): TypedComplex {
  const s = normalise(text);
  if (s === null) return fail('nan');
  if (s === '') return fail('empty');
  // polar: r∠θ
  const at = s.search(/[∠<]/);
  if (at >= 0) {
    const rt = s.slice(0, at), th = s.slice(at + 1);
    if (!rt || !th || /[∠<]/.test(th)) return fail('nan');
    return polar(rt, th);
  }
  // exponential: r·e^(iθ)
  const ex = s.match(/^(.*?)[*·×]?e\^(?:\((.+)\)|\{(.+)\}|(i.+))$/);
  if (ex) {
    const sx = ex[2] ?? ex[3] ?? ex[4];
    const sg = sx.startsWith('-') ? '-' : '', x = sx.replace(/^[+-]/, '');
    // i·θ with θ a single number: "iπ/3", "i/2" (½), "2πi/5", "1.2i"; "π/3i" is π/(3i) to some readers: refused
    let t0 = x.startsWith('i') ? x.slice(1).replace(/^[*·×]/, '') : x.includes('πi') ? x.replace('πi', 'π') : x.endsWith('i') && !x.includes('/') ? x.slice(0, -1).replace(/[*·×]$/, '') : null;
    // a real term beside iθ ("i+π/3", "i-1") is not an angle: refused, never read as one (Codex implementation review 1)
    if (t0 === null || t0.includes('i') || /^[+-]/.test(t0)) return fail('nan');
    if (t0.startsWith('/')) t0 = `1${t0}`;
    const th = sg + (t0 === '' ? '1' : t0); // e^(i) is e^{i·1}
    return polar(ex[1] === '' || ex[1] === '+' ? '1' : ex[1] === '-' ? '-1' : ex[1], th);
  }
  // rectangular: one or two terms, split at a + or − that is not a sign at the start or an exponent's (1e-3)
  const terms: string[] = [];
  let from = 0;
  for (let k = 1; k < s.length; k++) {
    if (s[k] !== '+' && s[k] !== '-') continue;
    if (s[k - 1] === 'e' && k >= 2 && /[\d.]/.test(s[k - 2])) continue;
    if (/[+\-*·×/∠^]/.test(s[k - 1])) continue;
    terms.push(s.slice(from, k));
    from = k;
  }
  terms.push(s.slice(from));
  if (terms.length > 2) return fail('nan');
  let re: TypedReal | null = null, im: TypedReal | null = null;
  for (const t of terms) {
    const sign = t.startsWith('-') ? '-' : '';
    const body = t.replace(/^[+-]/, '');
    const n = (body.match(/i/g) ?? []).length;
    if (n > 1) return fail('nan');
    if (n === 0) {
      if (re) return fail('nan');
      const v = typedReal(sign + body);
      if (typeof v === 'string') return fail(v);
      re = v;
      continue;
    }
    if (im) return fail('nan');
    // the coefficient before i ("2i", "√3/2i", "2·i") or after it ("i√3/2", "i/2" is 1/2)
    let coef = body.endsWith('i') ? body.slice(0, -1).replace(/[*·×]$/, '') : body.startsWith('i') ? body.slice(1).replace(/^[*·×]/, '') : null;
    if (coef === null) return fail('nan');
    if (coef.startsWith('/')) coef = `1${coef}`;
    const v = typedReal(sign + (coef === '' ? '1' : coef));
    if (typeof v === 'string') return fail(v);
    im = v;
  }
  const zero: TypedReal = { value: 0, q: Q.ZERO, sq: Q.ZERO };
  return { ok: true, form: 'rect', re: re ?? zero, im: im ?? zero };
}

function polar(rt: string, th: string): TypedComplex {
  const r = typedReal(rt);
  if (typeof r === 'string') return fail(r);
  if (r.value < 0) return fail('nan');
  const a = typedAngle(th);
  if (typeof a === 'string') return fail(a);
  return { ok: true, form: 'polar', r, angle: a };
}
