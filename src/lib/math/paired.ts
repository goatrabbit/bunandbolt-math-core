// Paired data kept in hundredths (whole numbers), and r worked out exactly: the sums about the means in BigInt, whether r
// is exactly 0 or ±1, and 1 − |r| without the cancellation of doubles. Written for H07 (correlation, design doc 改动 8,
// D-2) and moved here (2026-10-13) when H08 (linear regression) needed the same r.

export type XY = [number, number];

/** n times the sums of squares and products about the means, exactly: Cxx = nΣX² − (ΣX)² = n·Σ(X − X̄)², with ΣX, ΣY */
export interface PairedSums { n: bigint; sx: bigint; sy: bigint; Cxx: bigint; Cyy: bigint; Cxy: bigint }
export function pairedSums(hs: readonly XY[]): PairedSums {
  const n = BigInt(hs.length);
  let sx = 0n, sy = 0n, sxx = 0n, syy = 0n, sxy = 0n;
  for (const [x, y] of hs) { const X = BigInt(x), Y = BigInt(y); sx += X; sy += Y; sxx += X * X; syy += Y * Y; sxy += X * Y; }
  return { n, sx, sy, Cxx: n * sxx - sx * sx, Cyy: n * syy - sy * sy, Cxy: n * sxy - sx * sy };
}

/** r of data in hundredths, worked out from exact sums: whether it is exactly 0 or ±1, and δ = 1 − |r| without the
 *  cancellation of 1 − r in doubles ((0, 0), (999.99, 999.98), (1000, 999.99) has r = 1 in doubles, but
 *  1 − r ≈ 3.75 × 10⁻²¹). δ = (Q/D)/(1 + |r|) with D = CxxCyy, Q = D − Cxy². Null where r is undefined. */
export interface Fine { r: number; exact: 0 | 1 | -1 | null; delta: number }
export function fineR(hs: readonly XY[] | PairedSums): Fine | null {
  const s = Array.isArray(hs) ? pairedSums(hs as readonly XY[]) : (hs as PairedSums);
  if (s.n < 2n) return null;
  const { Cxx, Cyy, Cxy } = s;
  if (Cxx === 0n || Cyy === 0n) return null;
  if (Cxy === 0n) return { r: 0, exact: 0, delta: 1 };
  const D = Cxx * Cyy, Q = D - Cxy * Cxy, sign = Cxy > 0n ? 1 : -1;
  if (Q === 0n) return { r: sign, exact: sign, delta: 0 };
  // Cxx and Cyy convert to doubles with a relative error of at most 2⁻⁵³ (exactly up to 9·10¹⁵), so r and Q/D carry a
  // double's relative error
  const r = Number(Cxy) / Math.sqrt(Number(Cxx) * Number(Cyy)), delta = Number(Q) / Number(D) / (1 + Math.abs(r));
  return { r: Math.max(-1, Math.min(1, r)), exact: null, delta };
}

export const bgcd = (a: bigint, b: bigint): bigint => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
export function bsqrt(v: bigint): bigint {
  if (v < 2n) return v;
  let x = BigInt(Math.floor(Math.sqrt(Number(v))));
  while (x * x > v) x--;
  while ((x + 1n) * (x + 1n) <= v) x++;
  return x;
}

const minus = (s: string) => s.replace(/-/g, '−');
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
/** a value in scientific notation, two significant digits: “3.8 × 10⁻²¹” */
export function sci(v: number): string {
  const [m, e] = Math.abs(v).toExponential(1).split('e'), ex = Number(e);
  return `${v < 0 ? '−' : ''}${m} × 10${ex < 0 ? '⁻' : ''}${[...String(Math.abs(ex))].map((c) => SUP[Number(c)]).join('')}`;
}
/** r: “= 0”, “= 1”, “= −1” when exact (or `exactText`, an exact value the page knows otherwise: H07's ρ, H08's 0.9),
 *  else “≈0.816”, with more places when three would write a value that is not 0 or ±1 as 0.000 or ±1.000; within 10⁻⁶
 *  of ±1 from δ (“≈ 1 − 3.8 × 10⁻²¹”), within 10⁻⁶ of 0 in scientific notation */
export function rText(r: number, fine: Fine | null, exactText: string | null = null): string {
  if (exactText !== null) return `= ${minus(exactText)}`;
  if (fine?.exact != null) return `= ${minus(String(fine.exact))}`;
  if (fine && fine.delta < 1e-6) return `≈${r < 0 ? '−1 + ' : '1 − '}${sci(fine.delta)}`;
  if (Math.abs(r) < 1e-6) return `≈${sci(r)}`;
  for (let d = 3; d <= 8; d++) {
    const s = r.toFixed(d), v = Math.abs(Number(s));
    if (v !== 0 && v !== 1) return `≈${minus(s)}`;
  }
  return `≈${minus(r.toPrecision(10))}`;
}
/** a value with no exact form: three decimals (“≈1.206”), never 0 for a non-zero (below 10⁻⁶: “3.8 × 10⁻²¹”) */
export function approx3(v: number): string {
  if (v === 0) return '0';
  if (Math.abs(v) < 1e-6) return sci(v);
  for (let d = 3; d <= 8; d++) { const s = v.toFixed(d); if (Number(s) !== 0) return minus(s); }
  return sci(v);
}
