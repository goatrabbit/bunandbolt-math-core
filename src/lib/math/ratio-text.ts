// Exact ratios as a readout writes them (spec §5.3: one relation sign per value, “=” only when exact). A page that keeps
// its values as whole hundredths (H02's data, H05's μ, σ and cuts) gets exact fractions; this writes them “= 2.8” when
// the reduced denominator has only 2s and 5s (a finite decimal, written out by integers), else “≈9.78”. Moved here from
// H02 (2026-10-13) when H05 needed the same rules.

export const gcd = (a: number, b: number): number => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; };
const minus = (s: string) => s.replace('-', '−');
const trimZeros = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
/** num/den reduced, with the den positive */
export function reduce(num: number, den: number): [number, number] {
  const g = gcd(num, den) || 1, s = den < 0 ? -1 : 1;
  return [(s * num) / g, (s * den) / g];
}
/** a reduced denominator made only of 2s and 5s: num/den is a finite decimal, with this many places */
export function finitePlaces(den: number): number | null {
  let a = 0, b = 0;
  while (den % 2 === 0) { den /= 2; a++; }
  while (den % 5 === 0) { den /= 5; b++; }
  return den === 1 ? Math.max(a, b) : null;
}
/** a finite decimal num/den written out exactly, by integer long division in BigInt (no rounding, however many places:
 *  5242.89/5242.88 − 0 is 1.0000019073486328125, nineteen places, which a double would cut short) */
export function decimalText(num: number, den: number, places: number): string {
  const scaled = (BigInt(num) * 10n ** BigInt(places)) / BigInt(den); // exact: den divides num·10^places
  const neg = scaled < 0n, digits = (neg ? -scaled : scaled).toString().padStart(places + 1, '0');
  const s = places ? `${digits.slice(0, -places)}.${digits.slice(-places)}` : digits;
  return `${neg ? '−' : ''}${trimZeros(s)}`;
}
/** an approximation that never writes a non-zero as 0: two decimals from 1 up, else three significant digits */
export function approx(v: number): string {
  return minus(Math.abs(v) >= 1 ? v.toFixed(2) : Number(v.toPrecision(3)).toString());
}
/** The fraction num/den as a readout writes it: “= 2.8”, “= 0.00025” (a finite decimal, exact), else “≈9.78”. */
export function ratioText(num: number, den: number): string {
  const [p, q] = reduce(num, den), places = finitePlaces(q);
  return places !== null ? `= ${decimalText(p, q, places)}` : `≈${approx(p / q)}`;
}
/** a value kept in hundredths, written plainly: “5”, “9.4”, “−0.25” */
export const hundredthsText = (h: number) => decimalText(h, 100, 2);
