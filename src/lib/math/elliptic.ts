// The arithmetic–geometric mean and the complete elliptic integral of the first kind (the pendulum's period, G04).
//
// AGM(a, b): a, b → (a + b)/2, √(ab) converges quadratically to a common limit (Gauss); a handful of steps reach the
// last bit. K(k) = ∫₀^{π/2} dφ / √(1 − k² sin²φ) = π / (2·AGM(1, k′)) with k′ = √(1 − k²) (Abramowitz & Stegun 17.6).
// Near k = 1 the complementary modulus k′ carries the information and √(1 − k²) formed from k loses it, so ellipKc takes
// k′ itself — the pendulum knows it exactly as cos(θ₀/2).

/** The arithmetic–geometric mean of a, b ≥ 0 (NaN for a negative or non-finite argument; 0 when either is 0). */
export function agm(a: number, b: number): number {
  if (!(a >= 0 && b >= 0) || !Number.isFinite(a) || !Number.isFinite(b)) return NaN;
  if (a === 0 || b === 0) return 0;
  // AGM is homogeneous: work at scale 1 so √(ab) neither overflows nor underflows (Codex review 12)
  const M = Math.max(a, b);
  if (M !== 1) return M * agm(a / M, b / M);
  for (let i = 0; i < 64; i++) {
    const m = (a + b) / 2, g = Math.sqrt(a * b);
    if (m === a && g === b) break; // nothing moves any more
    if (Math.abs(m - g) <= 1e-16 * m) return (m + g) / 2;
    a = m;
    b = g;
  }
  return (a + b) / 2;
}

/** K from the complementary modulus k′ = √(1 − k²), 0 < k′ ≤ 1: π / (2·AGM(1, k′)). Infinity at k′ = 0. */
export function ellipKc(kc: number): number {
  if (!(kc >= 0 && kc <= 1)) return NaN;
  if (kc === 0) return Infinity;
  return Math.PI / (2 * agm(1, kc));
}

/** K(k), the complete elliptic integral of the first kind with modulus k, 0 ≤ k < 1. */
export function ellipK(k: number): number {
  if (!(Math.abs(k) <= 1)) return NaN;
  return ellipKc(Math.sqrt((1 - k) * (1 + k)));
}
