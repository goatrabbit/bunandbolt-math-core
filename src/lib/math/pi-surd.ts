// Exact readout values of the form p·π + (a + b√t) (p, a, b rational), and how the readouts write them. Lifted from
// C09's page code (2026-10-12) when C08 needed the same: a pen's coordinates at an exact parameter (the cycloid's
// x = t − sin t, the spiral's r = θ) carry a π part; trigonometric ones pass through one square root from the special
// angles' table (src/lib/math/angle.ts). Pure: no DOM. A value that does not fit this form is the page's to show as a
// rounded decimal, never a guessed exact one.
import { Q } from './rational';
import { Surd } from './surd';
import { exactValues, isSpecialK, piText, type Abs } from './angle';

const PI = Math.PI;

/** An exact real number p·π + s, s a surd. */
export interface Ex { pi: Q; s: Surd }
export const ex = (s: Surd, pi: Q = Q.ZERO): Ex => ({ pi, s });
export const exQ = (q: Q) => ex(Surd.of(q));
export const exValue = (e: Ex) => (Number(e.pi.n) * PI) / Number(e.pi.d) + e.s.toNumber();
export const exIsZero = (e: Ex) => e.pi.isZero() && e.s.a.isZero() && e.s.b.isZero();

const qText = (q: Q) => {
  const a = q.abs();
  return `${q.sign() < 0 ? '−' : ''}${a.d === 1n ? a.n : `${a.n}/${a.d}`}`;
};
/** "√3/2", "3√2", "√5" for b·√r with b > 0. */
const rootText = (b: Q, r: bigint) => `${b.n === 1n ? '' : b.n}√${r}${b.d === 1n ? '' : `/${b.d}`}`;
/** A surd as the readouts write it: "−√3/2", "1/2 + √2", "−99/101". */
export function surdText(s: Surd): string {
  if (s.b.isZero()) return qText(s.a);
  const root = rootText(s.b.abs(), s.root);
  if (s.a.isZero()) return `${s.b.sign() < 0 ? '−' : ''}${root}`;
  return `${qText(s.a)} ${s.b.sign() < 0 ? '−' : '+'} ${root}`;
}
/** p·π + s: "π", "π/2 − 1", "2π", "π/3 − √3/2". */
export function exText(e: Ex): string {
  if (e.pi.isZero()) return surdText(e.s);
  const p = piText(e.pi);
  if (e.s.a.isZero() && e.s.b.isZero()) return p;
  // the surd's leading sign goes between: π/2 − 1, π/3 − √3/2
  const lead = e.s.a.isZero() ? e.s.b.sign() : e.s.a.sign();
  return `${p} ${lead < 0 ? '−' : '+'} ${surdText(lead < 0 ? e.s.neg() : e.s)}`;
}

/** The table's absolute values of sin and cos as surds. */
const ABS_SURD: Partial<Record<Abs, Surd>> = {
  '0': Surd.of(0), '1/2': Surd.of(Q.of(1, 2)), '√2/2': Surd.sqrt(Q.of(1, 2))!, '√3/2': Surd.sqrt(Q.of(3, 4))!, '1': Surd.of(1),
};
/** sin and cos of q·π exactly when q·π is a special angle (a multiple of π/6 or π/4), else null. */
export function trigQ(q: Q): { sin: Surd; cos: Surd } | null {
  if (12n % q.d !== 0n) return null;
  const k = Number(q.n * (12n / q.d));
  if (!Number.isSafeInteger(k) || !isSpecialK(k)) return null;
  const v = exactValues(k);
  const one = (f: 'sin' | 'cos') => {
    const x = v[f];
    if (x.kind !== 'exact') return null;
    const s = ABS_SURD[x.abs];
    return s ? (x.sign < 0 ? s.neg() : s) : null;
  };
  const sin = one('sin'), cos = one('cos');
  return sin && cos ? { sin, cos } : null;
}
