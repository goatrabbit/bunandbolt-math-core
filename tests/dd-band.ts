// The certificate for the exact binomial band (design/pages/H12-law-of-large-numbers.md 数学; Codex design review
// 2026-10-07, finding 1): the same quantile walk as binomialBand (src/lib/math/sums.ts), in double-double arithmetic
// (about 32 significant digits; Dekker's products and Knuth's sums, no FMA needed). Every operation of the walk is
// then good to ~1e-31 relative, so after the 4·10⁵ operations of a walk to n = 10⁵ the probabilities it compares with
// 0.025 are good to well under 1e-20. Where this walk's decisions agree with the double walk's and stay more than
// 1e-20 from 0.025, the double band is the exact band. Imported by tests/math-sums.test.ts (a few p) and
// scripts/certify-binomial-band.mjs (all 99 p of the page's grid). No imports: node runs it as is.

type DD = [number, number];

const SPLIT = 134217729; // 2^27 + 1
function twoSum(a: number, b: number): DD {
  const s = a + b, bb = s - a;
  return [s, (a - (s - bb)) + (b - bb)];
}
function quick(a: number, b: number): DD {
  const s = a + b;
  return [s, b - (s - a)];
}
function twoProd(a: number, b: number): DD {
  const p = a * b;
  const ca = SPLIT * a, ah = ca - (ca - a), al = a - ah;
  const cb = SPLIT * b, bh = cb - (cb - b), bl = b - bh;
  return [p, ((ah * bh - p) + ah * bl + al * bh) + al * bl];
}
const add = (a: DD, b: DD): DD => { const [s, e] = twoSum(a[0], b[0]); return quick(s, e + a[1] + b[1]); };
const neg = (a: DD): DD => [-a[0], -a[1]];
const mul = (a: DD, b: DD): DD => { const [p, e] = twoProd(a[0], b[0]); return quick(p, e + a[0] * b[1] + a[1] * b[0]); };
const mulD = (a: DD, b: number): DD => { const [p, e] = twoProd(a[0], b); return quick(p, e + a[1] * b); };
function div(a: DD, b: DD): DD {
  const q1 = a[0] / b[0];
  let r = add(a, neg(mul(b, [q1, 0])));
  const q2 = r[0] / b[0];
  r = add(r, neg(mul(b, [q2, 0])));
  const q3 = r[0] / b[0];
  const s = quick(q1, q2);
  return add(s, [q3, 0]);
}
const divD = (a: DD, b: number) => div(a, [b, 0]);
const gt = (a: DD, b: DD) => a[0] > b[0] || (a[0] === b[0] && a[1] > b[1]);
const sub = (a: DD, b: DD) => add(a, neg(b));
const abs = (a: DD): DD => (a[0] < 0 || (a[0] === 0 && a[1] < 0) ? neg(a) : a);

/** The lower ends of the band of Binomial(m, P/100), m = 0 … M (the largest a with P(Xₘ < a) ≤ half), in double-double,
 *  with p = P/100 and half = num/den exactly as written; `margin` is the smallest |probability − half| over every
 *  comparison that settled an end (as a double). */
export function ddLowerWalk(M: number, P: number, num = 1, den = 40): { lo: Int32Array; margin: number } {
  const p = divD([P, 0], 100), q = divD([100 - P, 0], 100), r = div(p, q), half = divD([num, 0], den);
  const lo = new Int32Array(M + 1);
  let margin = Infinity;
  let j = 0, F: DD = [1, 0], f: DD = [1, 0];
  for (let m = 0; m <= M; m++) {
    if (m > 0) {
      F = sub(F, mul(p, f));
      f = divD(mulD(mul(f, q), m), m - j);
    }
    for (;;) {
      if (!gt(F, half)) {
        margin = Math.min(margin, abs(sub(half, F))[0]);
        f = divD(mul(mulD(f, m - j), r), j + 1);
        j++;
        F = add(F, f);
        continue;
      }
      if (j > 0 && gt(sub(F, f), half)) {
        margin = Math.min(margin, abs(sub(sub(F, f), half))[0]);
        F = sub(F, f);
        f = div(mulD(f, j), mulD(r, m - j + 1));
        j--;
        continue;
      }
      break;
    }
    margin = Math.min(margin, abs(sub(F, half))[0], j > 0 ? abs(sub(half, sub(F, f)))[0] : Infinity);
    lo[m] = j;
  }
  return { lo, margin };
}
