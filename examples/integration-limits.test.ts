// Example 3: a numerical integral reports some failures, and has limits.
//
// adaptiveSimpson returns a value, an error estimate and ok. ok: false means the method could not meet its tolerance
// (a non-finite sample, a kink or a near-singularity it cannot resolve within its depth budget); the value is then the
// best it has, and a page shows it as unconfirmed or not at all. ok: true is evidence, not proof: no sampling scheme
// can rule out aliasing at every frequency. Every branch is split a few times before it may stop, which catches slow
// waves but not fast ones. The pages use a closed form whenever they have one; quadrature is for what a reader types
// or draws, and for checks.
import { describe, expect, it } from 'vitest';
import { adaptiveSimpson } from '../src/lib/math/quad';

describe('adaptiveSimpson', () => {
  it('integrates a smooth function: ∫₀¹ x² dx = 1/3', () => {
    const r = adaptiveSimpson((x) => x * x, 0, 1);
    expect(r.ok).toBe(true);
    expect(Math.abs(r.value - 1 / 3)).toBeLessThan(1e-15);
  });

  it('splits before it stops, so cos 8πx on [0, 1] is not taken for the constant 1', () => {
    const f = (x: number) => Math.cos(8 * Math.PI * x);
    expect(Math.abs(adaptiveSimpson(f, 0, 1).value)).toBeLessThan(1e-10); // the integral is 0
    // without the minimum splits its first five samples are all 1, and it stops at once, confidently wrong
    const hasty = adaptiveSimpson(f, 0, 1, { minDepth: 0 });
    expect(hasty.ok).toBe(true);
    expect(Math.abs(hasty.value - 1)).toBeLessThan(1e-12);
  });

  it('is still fooled by a faster wave: cos 128πx on [0, 1] passes as 1', () => {
    // the 65 samples of the minimum splits all fall on peaks of the wave; nothing tells it apart from the constant 1
    const r = adaptiveSimpson((x) => Math.cos(128 * Math.PI * x), 0, 1);
    expect(r.ok).toBe(true);
    expect(Math.abs(r.value - 1)).toBeLessThan(1e-12); // the integral is 0
  });

  it('rejects an infinite endpoint sample, though ∫₀¹ dx/√x = 2 is finite', () => {
    const r = adaptiveSimpson((x) => 1 / Math.sqrt(x), 0, 1);
    expect(r.ok).toBe(false);
    expect(r.value).toBe(Infinity); // f(0) is infinite; this method does not handle the improper integral
  });

  it('flags a value it could not confirm: dx/√x from 10⁻¹² to 1', () => {
    const r = adaptiveSimpson((x) => 1 / Math.sqrt(x), 1e-12, 1);
    expect(r.ok).toBe(false); // the depth budget ran out next to the near-singularity
    expect(Math.abs(r.value - (2 - 2e-6))).toBeLessThan(1e-9); // close this time, but not certified
  });
});
