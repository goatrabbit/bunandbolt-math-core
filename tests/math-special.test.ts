import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  betaCdf, betaI, betaPdf, betaQuantile, betaSf, chi2Cdf, chi2Pdf, chi2Quantile, chi2Sf, erf, erfc, gamma, gammaCdf, gammaP,
  gammaPdf, gammaPQ, gammaQ, gammaQuantile, gammaSf, lgamma, normalCdf, normalPdf, normalQuantile, normalQuantileUpper, normalSf,
  normalTwoTail, tCdf, tPdf, tQuantile, tSf,
} from '../src/lib/math/special';
import { adaptiveSimpson } from '../src/lib/math/quad';

// Reference values: mpmath 1.3.0 at 50–60 significant digits (erf, erfc, ncdf, loggamma, gammainc, betainc, with findroot
// for quantiles), rounded to 17 digits. Classic table values (Abramowitz & Stegun tables 7.1, 26.7, 26.8) agree to the
// digits those tables print. Script values cite review/artifacts/page-directions/.

const COT_PI_OVER_2048 = 651.89813557739379; // cot(π/2048), mpmath

/** |got − want| ≤ tol·|want| (relative), or ≤ tol when want is 0 */
function rel(got: number, want: number, tol: number) {
  const err = want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want);
  expect(err, `got ${got}, want ${want}, relative error ${err}`).toBeLessThanOrEqual(tol);
}

describe('erf / erfc', () => {
  it('matches mpmath to ~1e-15 relative near the origin', () => {
    rel(erf(0.1), 0.11246291601828489, 2e-15);
    rel(erf(0.5), 0.52049987781304654, 2e-15);
    rel(erf(1), 0.84270079294971487, 2e-15);
    rel(erf(2), 0.99532226501895273, 2e-15);
    rel(erf(-0.3), -0.32862675945912743, 2e-15);
    rel(erf(1e-10), 1.1283791670955126e-10, 2e-15);
    expect(erf(0)).toBe(0);
  });
  it('keeps full relative accuracy far into the upper tail (no 1 − erf cancellation)', () => {
    rel(erfc(0.5), 0.47950012218695346, 1e-14);
    rel(erfc(1), 0.15729920705028513, 1e-14);
    rel(erfc(3), 2.2090496998585441e-5, 1e-14);
    rel(erfc(5), 1.5374597944280349e-12, 1e-14);
    rel(erfc(10), 2.0884875837625448e-45, 1e-14);
    rel(erfc(20), 5.3958656116079009e-176, 1e-13);
    rel(erfc(26.5), 2.2109076642637343e-307, 1e-12);
    rel(erfc(-1), 1.8427007929497149, 1e-15);
    expect(erfc(30)).toBe(0); // below the smallest subnormal
    expect(erfc(-30)).toBe(2);
  });
});

describe('normal distribution', () => {
  it('CDF and upper tail: relative accuracy in both tails', () => {
    rel(normalCdf(-1.96), 0.024997895148220434, 1e-14);
    rel(normalCdf(0.3), 0.61791142218895264, 1e-15);
    rel(normalCdf(-10), 7.6198530241605261e-24, 1e-13);
    rel(normalCdf(-37), 5.7255712225245768e-300, 1e-12);
    rel(normalSf(8.3), 5.2055697448902852e-17, 1e-13);
    rel(normalSf(15), 3.6709661993127509e-51, 1e-13);
    expect(normalCdf(0)).toBe(0.5);
    rel(normalTwoTail(1.959963984540054), 0.05, 1e-14);
    // H21: one-sided p-values Φ(−2), Φ(−5)
    rel(normalCdf(-2), 0.02275013194817932, 1e-13);
    rel(normalCdf(-5), 2.866515718791946e-7, 1e-12);
  });
  it('pdf integrates to the CDF', () => {
    const r = adaptiveSimpson(normalPdf, -40, 1.3, { tol: 1e-14 });
    rel(r.value, normalCdf(1.3), 1e-12);
  });
  it('quantile: z* values (H20) and the far tails', () => {
    rel(normalQuantile(0.975), 1.9599639845400542, 1e-15); // H20: z* 95% = 1.959963984540054
    rel(normalQuantile(0.995), 2.575829303548901, 1e-14); // H20: z* 99% = 2.5758293035489
    rel(normalQuantile(0.3), -0.52440051270804078, 1e-14);
    rel(normalQuantile(1e-10), -6.3613409024040562, 1e-14);
    rel(normalQuantile(1e-20), -9.2623400897984076, 1e-14);
    rel(normalQuantile(1e-300), -37.047096299361199, 1e-14);
    expect(normalQuantile(0.5)).toBe(0);
    expect(normalQuantile(0)).toBe(-Infinity);
    expect(normalQuantile(1)).toBe(Infinity);
    expect(normalQuantile(-0.1)).toBeNaN();
    // the upper-tail form gives what 1 − q can no longer hold
    rel(normalQuantileUpper(1e-20), 9.2623400897984076, 1e-14);
    rel(normalQuantileUpper(0.025), 1.9599639845400542, 1e-15);
    rel(normalQuantileUpper(0.7), -0.52440051270804078, 1e-14);
  });
  it('quantile inverts the CDF across the range', () => {
    for (let z = -37; z <= 8; z += 0.37) {
      const back = z < 0 ? normalQuantile(normalCdf(z)) : normalQuantileUpper(normalSf(z));
      expect(Math.abs(back - z)).toBeLessThan(1e-12 * Math.max(1, Math.abs(z)));
    }
  });
});

describe('gamma function', () => {
  it('lgamma and gamma', () => {
    // absolute error below 5e-15 for moderate x (so Γ = e^lgamma is good to 5e-15 relative), relative 1e-15 for large x
    const near = (got: number, want: number) => expect(Math.abs(got - want)).toBeLessThan(5e-15);
    near(lgamma(0.5), 0.57236494292470009);
    near(lgamma(1e-8), 18.420680738180209);
    near(lgamma(3.7), 1.4280723266653879);
    rel(lgamma(100), 359.1342053695754, 1e-15);
    rel(lgamma(1e5), 1051287.7089736569, 1e-15);
    near(lgamma(-0.5), 1.2655121234846454); // ln|Γ(−1/2)| = ln 2√π
    near(lgamma(-2.5), -0.056243716497674051);
    near(lgamma(1), 0);
    near(lgamma(2), 0);
    expect(lgamma(0)).toBe(Infinity);
    expect(lgamma(-3)).toBe(Infinity);
    rel(gamma(5), 24, 1e-14);
    rel(gamma(4.5), 11.631728396567449, 1e-14);
    rel(gamma(0.1), 9.5135076986687318, 1e-14);
    rel(gamma(-1.5), 2.3632718012073547, 1e-14);
    rel(gamma(170.5), 5.5620924145599996e305, 1e-12);
  });
  it('regularized incomplete gamma: P and Q, each accurate in its own tail', () => {
    rel(gammaP(1, 1), 0.63212055882855768, 1e-14);
    rel(gammaP(0.5, 2), 0.95449973610364159, 1e-14); // = erf(√2)
    rel(gammaP(10, 5), 0.031828057306204812, 1e-13);
    rel(gammaP(100, 90), 0.15822098918643017, 1e-12);
    rel(gammaP(3, 0.01), 1.6542165280748768e-7, 1e-13);
    rel(gammaQ(5, 100), 1.6139305336977305e-37, 1e-12);
    rel(gammaQ(30, 60), 6.8762649687320967e-6, 1e-12);
    rel(gammaQ(0.5, 30), 9.4857375710738484e-15, 1e-12); // = erfc(√30)
    rel(gammaQ(1000, 1100), 0.0010593232539299773, 1e-12);
    // very large a: the prefactor xᵃe⁻ˣ/Γ(a) is built without cancelling logarithms
    rel(gammaQ(1e4, 1.01e4), 0.15865124955282038, 1e-12);
    rel(gammaP(1e5, 99000), 0.00075741992117476797, 1e-11);
    const [p, q] = gammaPQ(7, 7);
    expect(Math.abs(p + q - 1)).toBeLessThan(1e-15);
  });
});

describe('regularized incomplete beta', () => {
  it('matches mpmath, both I and 1 − I', () => {
    rel(betaI(0.5, 2, 3)[0], 0.6875, 1e-15);
    rel(betaI(0.3, 0.5, 0.5)[0], 0.36901011956554538, 1e-14); // = (2/π) asin √0.3
    rel(betaI(0.999, 50, 2)[0], 0.99876590960688292, 1e-14);
    rel(betaI(0.999, 50, 2)[1], 0.0012340903931170825, 1e-12);
    rel(betaI(0.01, 5, 0.5)[0], 2.4712578086395444e-11, 1e-12);
    rel(betaI(0.3, 200, 300)[0], 1.0496985243292976e-6, 1e-10);
    rel(betaI(0.5, 30, 10)[1], 0.99946749018272385, 1e-13);
    expect(betaI(0, 2, 3)).toEqual([0, 1]);
    expect(betaI(1, 2, 3)).toEqual([1, 0]);
  });
});

describe('Student t', () => {
  it('CDF and upper tail', () => {
    rel(tCdf(2, 1), 0.85241638234956673, 1e-14); // 1/2 + atan(2)/π
    rel(tCdf(-40, 3), 1.7190340394579264e-5, 1e-12);
    rel(tCdf(1.5, 2.5), 0.87608177345685187, 1e-13);
    rel(tCdf(0.5, 7), 0.68379643215535789, 1e-13);
    rel(tSf(10, 5), 8.5473787871481795e-5, 1e-12);
    rel(tCdf(-10, 5), 8.5473787871481795e-5, 1e-12);
    expect(tCdf(0, 4)).toBe(0.5);
    // large ν: no digits lost to the size of ν
    rel(tSf(1.96, 1e6), 0.024998033792634893, 1e-12);
    rel(tCdf(1.96, 1e6), 0.97500196620736511, 1e-14);
    rel(tSf(2.5, 1000), 0.0062892839005453984, 1e-12);
    rel(tCdf(-3, 1e4), 0.0013532240949988331, 1e-12);
    rel(tCdf(0.001, 1e6), 0.50039894211417546, 1e-14);
    // H20: 2·F_t4(1.96) − 1, the coverage of x̄ ± 1.96·s/√5. mpmath gives 0.87844536014981643; H20 prints
    // 0.8784403402 (its own t CDF, checked only to ±0.0005), so the script's 6th digit is not a reference.
    rel(2 * tCdf(1.96, 4) - 1, 0.87844536014981643, 1e-13);
    expect(Math.abs(2 * tCdf(1.96, 4) - 1 - 0.878)).toBeLessThan(0.0005);
  });
  it('quantile over the whole control range, including n = 2 at 99.9%', () => {
    // H20: t*(0.9995, 1) = 636.6192487687135. The double 0.9995 lies 5.5e-17 above 0.9995, which alone moves this
    // quantile by 1.1e-13 relative; for exactly representable p the Cauchy branch is exact to rounding.
    rel(tQuantile(0.9995, 1), 636.61924876871962, 2e-13);
    rel(tQuantile(0.9995, 1), 636.61924876878972983, 2e-16); // mpmath with the double 0.9995 itself
    rel(tQuantile(1 - 2 ** -11, 1), COT_PI_OVER_2048, 2e-16);
    rel(tQuantile(1e-300, 1), -1 / (Math.PI * 1e-300), 1e-15); // Cauchy far tail: −cot(πp) ≈ −1/(πp)
    expect(tQuantile(0.5, 7)).toBe(0);
    rel(tQuantile(0.975, 4), 2.7764451051977944, 1e-13); // H20: 2.7764451051977854
    rel(tQuantile(0.995, 2), 9.9248432009182931, 1e-13);
    rel(tQuantile(0.9, 30), 1.3104150253913956, 1e-13);
    rel(tQuantile(0.975, 1e6), 1.959966356814107, 1e-13);
    rel(tQuantile(1e-12, 3), -10331.108244292486, 1e-11);
    // ν = 2 has the closed form (2p − 1)/√(2p(1 − p))
    for (const p of [0.6, 0.9, 0.99, 0.9999]) rel(tQuantile(p, 2), (2 * p - 1) / Math.sqrt(2 * p * (1 - p)), 1e-12);
    // H20: coverage 2·F_t1(t*) − 1 = 0.999 at that t*
    rel(2 * tCdf(tQuantile(0.9995, 1), 1) - 1, 0.999, 1e-13);
  });
  it('pdf integrates to the CDF, for small and large ν', () => {
    for (const nu of [3, 50, 1e6]) {
      const r = adaptiveSimpson((t) => tPdf(t, nu), -1, 2.5, { tol: 1e-13 });
      rel(r.value, tCdf(2.5, nu) - tCdf(-1, nu), 1e-11);
    }
    rel(tPdf(0, 1), 1 / Math.PI, 1e-14);
    rel(tPdf(1.3, 1e8), normalPdf(1.3), 1e-7); // ν → ∞: the normal density, to O(1/ν)
  });
});

describe('chi-square, gamma and beta distributions', () => {
  it('chi-square CDF, tail and quantile', () => {
    rel(chi2Quantile(0.95, 1), 3.841458820694126, 1e-13);
    rel(chi2Quantile(0.95, 10), 18.307038053275147, 1e-13);
    rel(chi2Quantile(0.001, 2), 0.002001000667167067, 1e-12);
    rel(chi2Quantile(0.5, 100), 99.334129235988456, 1e-13);
    rel(chi2Quantile(1e-8, 3), 1.1223330973050008e-5, 1e-10);
    rel(chi2Sf(200, 10), 1.6139305336977353e-37, 1e-12);
    rel(chi2Cdf(3.841458820694126, 1), 0.95, 1e-14);
    expect(chi2Cdf(0, 3)).toBe(0);
    expect(chi2Sf(-1, 3)).toBe(1);
    const r = adaptiveSimpson((x) => chi2Pdf(x, 5), 0, 7, { tol: 1e-13 });
    rel(r.value, chi2Cdf(7, 5), 1e-11);
  });
  it('gamma distribution (shape k, scale θ)', () => {
    rel(gammaQuantile(0.5, 3, 2), 5.3481206274471206, 1e-13); // = χ² median with 6 degrees of freedom
    rel(gammaCdf(5.3481206274471206, 3, 2), 0.5, 1e-14);
    rel(gammaSf(4, 1, 2), Math.exp(-2), 1e-14);
    const r = adaptiveSimpson((x) => gammaPdf(x, 2.5, 1.5), 0, 4, { tol: 1e-13 });
    rel(r.value, gammaCdf(4, 2.5, 1.5), 1e-11);
  });
  it('beta distribution', () => {
    rel(betaQuantile(0.025, 0.5, 0.5), 0.0015413331334360119, 1e-12); // sin²(π·0.025/2), the arcsine law
    rel(betaQuantile(0.5, 2, 3), 0.38572756813238955, 1e-13);
    rel(betaQuantile(0.3, 0.1, 0.1), 0.0052148702537342796, 1e-11);
    rel(betaQuantile(1e-10, 2, 5), 2.581997786411184e-6, 1e-10);
    rel(betaCdf(0.5, 2, 3) + betaSf(0.5, 2, 3), 1, 1e-15);
    const r = adaptiveSimpson((x) => betaPdf(x, 2, 3), 0, 0.7, { tol: 1e-13 });
    rel(r.value, betaCdf(0.7, 2, 3), 1e-11);
  });
  it('quantiles invert their CDFs at tail probabilities', () => {
    for (const p of [1e-12, 1e-6, 0.01, 0.5, 0.99, 1 - 1e-9]) {
      rel(tCdf(tQuantile(p, 7), 7), p, 1e-10);
      rel(chi2Cdf(chi2Quantile(p, 4), 4), p, 1e-10);
      rel(betaCdf(betaQuantile(p, 3, 0.7), 3, 0.7), p, 1e-10);
    }
  });
});

describe('incomplete gamma and beta on mpmath grids', () => {
  // generated by review/artifacts/math-lib/gamma_grid.py and beta_grid.py; null = no trustworthy reference for that side.
  // The bound: 1e-12 relative (1e-11 when a beta parameter is below 0.1), or 8 units of the function's own sensitivity
  // to the last bit of x, κ·2⁻⁵³ with κ = |d ln F / d ln x| — whichever is larger. Far in the tail of a huge shape the
  // value is that sensitive to x itself (κ ≈ 25 000 at a = 10⁷, eight SDs out); no method can do better from x.
  const load = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
  const U = 2 ** -53;
  const kappa = (f: (x: number) => number, x: number) => {
    const h = 1e-7, v = f(x), w = f(x * (1 + h));
    return v > 0 && w > 0 ? Math.abs(Math.log(w) - Math.log(v)) / h : 0;
  };
  const within = (got: number, want: string | null, base: number, k: number, label: string) => {
    if (want === null) return;
    const err = Math.abs(got - Number(want)) / Number(want);
    expect(err, `${label}: got ${got}, want ${want}`).toBeLessThanOrEqual(Math.max(base, 8 * U * k));
  };
  it('P and Q, both tails: shapes from 1e-310 to 1e7', () => {
    const grid = load('gamma-mpmath.json') as [number, number, string | null, string | null][];
    expect(grid.length).toBeGreaterThan(250);
    for (const [a, x, P, Q] of grid) {
      const [p, q] = gammaPQ(a, x);
      within(p, P, 1e-12, kappa((t) => gammaPQ(a, t)[0], x), `P(${a}, ${x})`);
      within(q, Q, 1e-12, kappa((t) => gammaPQ(a, t)[1], x), `Q(${a}, ${x})`);
    }
  });
  it('I and 1 − I, both tails: a, b from 1e-20 to 5·10⁵, points at the switch between the two sides', () => {
    const grid = load('beta-mpmath.json') as [number, number, number, string | null, string | null][];
    expect(grid.length).toBeGreaterThan(350);
    for (const [a, b, x, I, J] of grid) {
      const [i, j] = betaI(x, a, b);
      const base = Math.min(a, b) < 0.1 ? 1e-11 : 1e-12;
      within(i, I, base, kappa((t) => betaI(t, a, b)[0], x), `I(${x}; ${a}, ${b})`);
      within(j, J, base, kappa((t) => betaI(t, a, b)[1], x), `1 − I(${x}; ${a}, ${b})`);
    }
  });
});

describe('Codex review 2026-10-07 (review/2026-10-07-math-lib-codex.md): regressions', () => {
  it('#1 the t tail keeps its digits when 1 − (the other side) would round to 0', () => {
    rel(tSf(10, 1e6), 7.639305384089125e-24, 1e-13);
    rel(tCdf(-10, 1e6), 7.639305384089125e-24, 1e-13);
  });
  it('#3 |t| beyond 1.3e154 (t² overflows): tails and the quantiles that land there', () => {
    rel(tSf(1e155, 0.3), 1.10521832976946e-47, 1e-13);
    rel(tQuantile(1e-50, 0.3), -1.395812007068317e165, 1e-12);
    rel(tPdf(1e120, 1), 1e-240 / Math.PI, 1e-13); // the ln(1 + t²/ν) branch for huge t
  });
  it('#4 normal quantiles of subnormal probabilities', () => {
    rel(normalQuantile(1e-315), -37.96730035106736, 1e-14);
    rel(normalQuantileUpper(1e-315), 37.96730035106736, 1e-14);
    rel(normalQuantile(Number.MIN_VALUE), -38.467405617144346, 1e-12); // mpmath, 60 digits
  });
  it('#5 a subnormal x keeps its size in the beta prefactor', () => {
    rel(betaI(Number.MIN_VALUE, 0.1, 0.1)[0], 2.369134002591342e-33, 1e-12);
  });
  it('#6 a moderate a next to a huge b, just below the switch point', () => {
    rel(betaI(0.0002019588004047174, 100, 500000)[1], 0.44711957012863797, 1e-12);
  });
  it('#7 Q for a tiny shape is computed directly, not as 1 − P', () => {
    rel(gammaQ(1e-6, 1), 2.1938415887050923e-7, 1e-12);
  });
  it('#8 invalid parameters give NaN before the support test', () => {
    expect(gammaCdf(1, 2, -1)).toBeNaN();
    expect(gammaSf(1, 2, 0)).toBeNaN();
    expect(chi2Cdf(1, 0)).toBeNaN();
    expect(chi2Sf(-1, -2)).toBeNaN();
  });
  it('lgamma is exact relative to its size next to its zeros at 1 and 2', () => {
    rel(lgamma(1 + 1e-10), -5.772157125783244e-11, 1e-14);
    rel(lgamma(2 - 1e-9), -4.2278436975733279e-10, 1e-14);
    rel(lgamma(1.5), -0.12078223763524522, 1e-15);
    rel(lgamma(0.75), 0.20328095143129537, 1e-15);
  });
});

describe('Codex re-review 2026-10-07: the corners, fixed or refused', () => {
  it('lgamma against mpmath from 1e-310 to 1e305: 5e-15 absolute where |ln Γ| < 1, else 1e-15 relative', () => {
    const grid = JSON.parse(readFileSync(new URL('./fixtures/lgamma-mpmath.json', import.meta.url), 'utf8')) as [number, string][];
    expect(grid.length).toBeGreaterThan(250);
    for (const [x, v] of grid) {
      const want = Number(v), err = Math.abs(lgamma(x) - want);
      expect(err, `lgamma(${x})`).toBeLessThanOrEqual(Math.abs(want) < 1 ? 5e-15 : 1e-15 * Math.abs(want));
    }
  });
  it('#3 a quantile near the largest double, and one beyond it', () => {
    rel(tQuantile(tSf(1e308, 0.3), 0.3), -1e308, 1e-12);
    expect(tQuantile(1e-120, 0.3)).toBe(-Infinity); // beyond ±1.8e308
  });
  it('#7 a shape of 1e-310: the subnormal answer, to the digits a subnormal has', () => {
    rel(gammaQ(1e-310, 1), 2.1938393439552e-311, 1e-11);
    expect(gammaP(1e-310, 1)).toBe(1);
  });
  it('#8 infinite shape, scale or degrees of freedom are refused', () => {
    for (const v of [gammaCdf(1, 2, Infinity), gammaCdf(1, Infinity), chi2Cdf(0, Infinity), gammaPQ(Infinity, 1)[0], tCdf(1, Infinity),
      betaI(0.5, Infinity, 1)[0], tPdf(0, Infinity), betaQuantile(0.5, 2, Infinity), gammaQuantile(0.5, 1, Infinity)]) expect(v).toBeNaN();
  });
  it('a huge shape whose series exhausts its budget is refused, not truncated', () => {
    expect(gammaP(1e12, 1e12)).toBeNaN();
    expect(gammaPQ(1e7, 1e7 + 5000)[0]).toBeGreaterThan(0.9); // inside the budget: computed (the grid checks its digits)
  });
  it('a tiny beta parameter: closed forms for b = 1 and a = 1, a refusal where neither expansion converges', () => {
    const [i, j] = betaI(0.1, 1e-20, 1);
    expect(i).toBe(1);
    rel(j, 2.302585092994046e-20, 1e-14); // −expm1(1e-20·ln 0.1)
    rel(betaI(0.3, 1, 1e-18)[0], 3.566749439387324e-19, 1e-14); // −expm1(1e-18·ln 0.7)
    expect(betaI(2e-11, 1e-20, 3)).toEqual([NaN, NaN]);
  });
  it('quantiles end on the nearer of two adjacent doubles', () => {
    expect(betaQuantile(Number.MIN_VALUE, 1, 1)).toBe(Number.MIN_VALUE);
    expect(betaQuantile(0.25, 1, 1)).toBe(0.25);
  });
});

describe('Codex third review 2026-10-07', () => {
  it('a usable first beta tail is kept when the other expansion does not converge', () => {
    rel(tSf(1024, 0.01), 0.4527666551809315454, 1e-12); // mpmath, 80 digits
    rel(tQuantile(0.99, 0.01), 3.9604401371520978433e168, 1e-10); // κ = 1/ν: the quantile magnifies p's error 100×
  });
  it('quantiles of subnormal probabilities land on the plateau of the computed CDF', () => {
    const r = betaQuantile(Number.MIN_VALUE, 2, 1); // F(x) = x², exact answer 2⁻⁵³⁷ ≈ 2.2e-162
    expect(r).toBeLessThanOrEqual(2 ** -537);
    expect(betaCdf(r, 2, 1)).toBeLessThanOrEqual(Number.MIN_VALUE);
    rel(betaQuantile(1e-300, 2, 1), 1e-150, 1e-15); // a normal p: full precision
  });
});
