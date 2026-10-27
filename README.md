# Bun & Bolt Math: calculation core

The calculation code behind [Bun & Bolt Math](https://bunandbolt.com/math/), an interactive visual dictionary of mathematics, together with the tests that check it.

The pages show "=" only for values they know exactly and "≈" for the rest, and they decide questions such as "is this matrix singular?" or "is log₁₀ 1000 exactly 3?" without trusting a floating-point tolerance. This repository is where you can see how, and check it.

## What this is, and what it isn't

- **A snapshot of the site's shared calculation library**, exported unchanged from the site's source. [EXPORT.json](EXPORT.json) names the source commit and gives a SHA-256 for every other file.
- **Not a general-purpose package.** It isn't published on npm and has no stable API: the functions are shaped by the pages that use them and may change in any snapshot.
- **Not the whole site.** Each page's own calculations, the drawing, the controls and the explanations aren't included. Passing tests here doesn't mean every picture and sentence on the site has been verified, and tests we wrote ourselves aren't an independent audit.

## Run the tests

Node.js 22.12+, 24 or 26+:

```sh
npm ci
npm test
```

## Three examples

Each example is a test file in [examples/](examples/), so `npm test` runs it.

**A typed 0.1 is one tenth** ([exact-input.test.ts](examples/exact-input.test.ts)). Typed text is read with `parseExact`, so `0.1` becomes 1/10 rather than the nearest double, 3602879701896397 / 2⁵⁵. Pages make their exact decisions about typed decimals and fractions (equal or not, zero or not, which case) from that rational; evaluating a typed formula uses doubles.

```ts
parseExact('0.1')!.add(parseExact('0.2')!).eq(parseExact('0.3')!); // true (0.1 + 0.2 === 0.3 is false)
```

**A singular matrix stays singular** ([singular-matrix.test.ts](examples/singular-matrix.test.ts)). The rows of [[0.1, 0.7], [0.3, 2.1]] are proportional. As doubles the entries are slightly different numbers, so the determinant is about 2.8e-17 and the system has a "unique" solution. Read as typed, the determinant is exactly 0, and the system gets its true case.

```ts
det2([0.1, 0.7, 0.3, 2.1]);                     // 2.7755575615628914e-17
det2Q(['0.1', '0.7', '0.3', '2.1'].map((s) => parseExact(s)!) as Q2).isZero(); // true
```

**A numerical integral reports some failures, and has limits** ([integration-limits.test.ts](examples/integration-limits.test.ts)). `adaptiveSimpson` returns a value, an error estimate and `ok`. An infinite sample at an endpoint, or a depth budget run out next to a near-singularity, gives `ok: false`. Every branch is split a few times before it may stop, so cos 8πx on [0, 1] isn't taken for the constant 1. But `ok: true` is not a guarantee: cos 128πx happens to be 1 at every sample the method takes, and it reports the integral as 1 with no error. The true value is 0. The pages use a closed form whenever they have one.

## Modules

All in `src/lib/math/` unless noted. Each file begins with a comment on what it computes and where its limits are.

| Module | What it does | Exact or numerical |
|---|---|---|
| `rational.ts` | Rational numbers on BigInt; `parseExact` for typed text | Exact |
| `surd.ts` | Numbers a + b√t | Exact |
| `pi-surd.ts` | Numbers p·π + (a + b√t), and how readouts write them | Exact |
| `powers.ts` | Positive reals e^a·∏gᵢ^rᵢ; decides whether log_b x is rational, and which rational | Exact |
| `precise.ts` | Rigorous rational enclosures of π and of logarithms | Exact bounds |
| `ratio-text.ts` | Writes an exact ratio as "= 2.8" when it is a finite decimal, otherwise "≈9.78" | Exact finite decimals, otherwise rounded and labelled |
| `cyclotomic.ts` | Exact zero test for sums of sines and cosines of rational-degree angles | Exact |
| `angle.ts` | Angles as exact rational multiples of π or as decimal radians; special values from a declared table | Both, labelled |
| `complex.ts` | Complex numbers, with whatever is known exactly kept beside the double | Both, labelled |
| `linalg.ts` | 2×2 and 3×3 determinants, inverses, solves, eigenvalues, projections, least squares; 4D rotations | Exact rational predicates and results; doubles for drawing |
| `paired.ts` | The correlation r of paired data kept in hundredths | Exact sums and tests for r = 0 or ±1; other values in doubles |
| `expr.ts` | Safe parser, compiler and differentiator for typed formulas (no `eval`); errors as codes with positions | Symbolic parsing, evaluation in doubles, some exact rational checks |
| `damped.ts` | The damped oscillator in closed form; its regime is decided exactly | Closed forms in doubles |
| `driven.ts` | The driven damped oscillator (resonance) in closed form | Closed-form motion in doubles; numerical energy integrals and zeros |
| `elliptic.ts` | Arithmetic–geometric mean and the complete elliptic integral K | Numerical |
| `fft.ts` | Radix-2 fast Fourier transform | Numerical |
| `cpoly.ts` | Complex polynomials: evaluation, Taylor coefficients, all roots (Durand–Kerner), Smale's radius | Numerical |
| `roots.ts` | Root finding from a bracket; oscillating functions lobe by lobe | Numerical |
| `quad.ts` | Adaptive Simpson, Gauss–Legendre, disc and rectangle integrals, with error estimates | Numerical |
| `ode.ts` | Euler, RK4 and Dormand–Prince 5(4), which stops and says why | Numerical |
| `trajectory.ts` | An ODE solution you can ask for at any time (dense output), with crossings | Numerical |
| `maps2d.ts` | Iterated maps of the plane (Hénon, Lozi, Ikeda): Lyapunov exponents, periods, classification | Mostly numerical, labelled |
| `ifs.ts` | Iterated function systems in the plane (the chaos game) | Numerical |
| `special.ts` | erf, Φ and its inverse, gamma, incomplete gamma and beta; normal, t, χ², gamma and beta distributions | Numerical |
| `sampling.ts` | Sampling distributions of common statistics | Closed forms or quadrature, labelled; evaluated in doubles |
| `sums.ts` | Distributions of sums and means of independent draws | Finite sums, closed forms or numerical inversion, labelled; evaluated in doubles |
| `rng.ts` | The seeded generator and samplers of the statistics pages (versioned: a seed reproduces a run) | Deterministic |
| `src/lib/ui/typed.ts` | What a number field accepts: decimals, fractions, multiples of π, roots, e | Reads a double, with exact rational and symbolic readings alongside |

## Tests and references

`tests/` holds the site's tests for these modules. Reference values come from mpmath (computed at 40–60 significant digits), closed forms and published tables (Abramowitz & Stegun). The three mpmath grids in `tests/fixtures/` are generated by the scripts in [reference/mpmath/](reference/mpmath/).

The integer PRNG stream is bit-reproducible. Transformed samples can differ in their last floating-point bits across JavaScript engines; for example, [Math.log is implementation-approximated](https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math.log).

Not covered by a test in this repository: `ifs.ts`, `maps2d.ts`, `paired.ts`, `pi-surd.ts` and `ratio-text.ts`. They are exercised through the site's page tests, which depend on page code and aren't part of this snapshot.

Comments mention the site's page codes (C01, H19 …) and design notes such as `design/pages/…`. Those belong to the site's private source and aren't included. The mpmath scripts that comments place in `review/artifacts/math-lib/` are in `reference/mpmath/` here.

## Reporting a problem

Please open an issue with the input, the page link and the snapshot commit from EXPORT.json. Fixes are made in the site's source and arrive with the next snapshot, so we read pull requests as reports rather than merging them.

## Licence

The code in this repository is under the [MIT licence](LICENSE). The text, images and videos of Bun & Bolt Math are separately licensed under CC BY 4.0: see [Using our work](https://bunandbolt.com/privacy/#using-our-work). No trademark rights are granted in the names and logos Bun & Bolt and Bun & Bolt Math.

## 中文说明

这里是[Bun & Bolt Math](https://bunandbolt.com/zh/math/)网站计算部分的源代码和测试，从网站源码原样导出。每份快照对应的网站源码版本和文件校验值见EXPORT.json。

这不是通用的数学软件包：没有发布到npm，也不承诺接口稳定。各页面自己的计算、绘图、控件和讲解不在这里。这里的测试通过，不代表网站上每张图、每句话都已核对；自己写的测试也不等于第三方审核。

运行：Node.js 22.12+、24或26+，执行`npm ci`和`npm test`。三个示例在`examples/`：键入的0.1按十分之一读；奇异矩阵不会因为浮点误差被当成可逆；数值积分能报告一部分失败，但“可信”标记不是保证，示例里也给出了它被骗过的情形。

整数算法生成的均匀随机数流可逐位复现；经过Math.log等函数变换的样本，在不同JavaScript引擎间可能有浮点末位差异。

发现问题请开issue，写明输入、页面链接和快照版本。修改先进网站源码，再随下一份快照更新，所以拉取请求按问题报告处理，不直接合并。代码采用MIT许可；网站的文字、图片和视频另行采用CC BY 4.0许可；名称和标识不授予商标权。
