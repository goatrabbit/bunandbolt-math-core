import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { compile, dependsOn, diagnose, differentiate, exactDerivative, exactValue, parse, poleFactors, realPow, smoothAt, toTex, toText, type ErrorCode, type Node } from '../src/lib/math/expr';
import { Q } from '../src/lib/math/rational';

const ast = (s: string, vars?: string[]): Node => {
  const p = parse(s, { vars });
  if (!p.ok) throw new Error(`${s}: ${p.error.code} at ${p.error.at}`);
  return p.ast;
};
const val = (s: string, env: Record<string, number> = {}) => compile(ast(s))(env);
const code = (s: string, vars?: string[]): ErrorCode | 'ok' => {
  const p = parse(s, { vars });
  return p.ok ? 'ok' : p.error.code;
};

describe('parse and compile', () => {
  it('numbers, constants, operators and precedence', () => {
    expect(val('1 + 2*3')).toBe(7);
    expect(val('2^3^2')).toBe(512); // right-associative
    expect(val('-x^2', { x: 3 })).toBe(-9); // −(x²)
    expect(val('(-x)^2', { x: 3 })).toBe(9);
    expect(val('2^-1')).toBe(0.5);
    expect(val('2**3')).toBe(8);
    expect(val('7 − 2 × 3 ÷ 4')).toBe(5.5);
    expect(val('1e-3 + .5 + 2.')).toBe(2.501);
    expect(val('π')).toBe(Math.PI);
    expect(val('pi/2 + e')).toBe(Math.PI / 2 + Math.E);
    expect(val('[x + 1]·2', { x: 1 })).toBe(4);
  });
  it('every function, with log = base 10 and ln = natural', () => {
    const x = 0.3;
    const pairs: [string, number][] = [
      ['sin(x)', Math.sin(x)], ['cos(x)', Math.cos(x)], ['tan(x)', Math.tan(x)], ['asin(x)', Math.asin(x)],
      ['acos(x)', Math.acos(x)], ['atan(x)', Math.atan(x)], ['arcsin(x)', Math.asin(x)], ['sinh(x)', Math.sinh(x)],
      ['cosh(x)', Math.cosh(x)], ['tanh(x)', Math.tanh(x)], ['exp(x)', Math.exp(x)], ['ln(x)', Math.log(x)],
      ['log(x)', Math.log10(x)], ['sqrt(x)', Math.sqrt(x)], ['cbrt(x)', Math.cbrt(x)], ['abs(-x)', x], ['sign(-x)', -1],
      ['floor(x + 2)', 2], ['min(x, 1)', x], ['max(x, 1)', 1], ['pow(x, 2)', x * x], ['√x', Math.sqrt(x)], ['sgn(x)', 1],
    ];
    for (const [s, want] of pairs) expect(val(s, { x })).toBe(want);
  });
  it('implicit multiplication where it cannot be misread', () => {
    const env = { x: 2, y: 3, t: 5 };
    const ok: [string, number][] = [
      ['2x', 4], ['2(x + 1)', 6], ['2 sin(x)', 2 * Math.sin(2)], ['x(y + 1)', 8], ['(x)(y)', 6], ['xy', 6], ['xyt', 30],
      ['πx', 2 * Math.PI], ['2x^2', 8], ['x²y', 12], ['e^(2x)y', Math.exp(4) * 3], ['3√x', 3 * Math.SQRT2], ['xe', 2 * Math.E],
      ['2x/3', 4 / 3], ['(1/2)x', 1], ['1/(2x)', 0.25],
    ];
    for (const [s, want] of ok) expect(val(s, env), s).toBeCloseTo(want, 13);
  });
  it('ambiguous or malformed input returns a code with a position, never a guess', () => {
    const bad: [string, ErrorCode][] = [
      ['', 'empty'], ['   ', 'empty'], ['1/2x', 'ambiguous-division'], ['1/2(x + 1)', 'ambiguous-division'],
      ['e^2x', 'ambiguous-power'], ['x^2 y', 'ambiguous-power'], ['√x^2', 'ambiguous-power'], ['x²^3', 'ambiguous-power'],
      ['x2', 'unexpected'], ['2 3', 'unexpected'], ['1, 2', 'unexpected'], ['sin x', 'function-needs-paren'],
      ['sinx', 'function-needs-paren'], ['lnx', 'function-needs-paren'], ['min(x)', 'wrong-arity'], ['sin(x, y)', 'wrong-arity'],
      ['2 +', 'missing-operand'], ['* x', 'missing-operand'], ['x^', 'missing-operand'], ['√', 'missing-operand'],
      ['(x + 1', 'missing-paren'], ['sin(x', 'missing-paren'], ['x + 1)', 'extra-paren'], ['()', 'empty-parens'],
      ['sin()', 'empty-parens'], ['1.2.3', 'bad-number'], ['2e-', 'bad-number'], ['x $ 2', 'bad-character'],
      ['foo(x)', 'unknown-name'], ['a', 'unknown-variable'], ['constructor', 'unknown-name'], ['constructor(x)', 'unknown-name'],
      ['toString(x)', 'unknown-name'], ['__proto__', 'bad-character'],
    ];
    for (const [s, c] of bad) expect(code(s), s).toBe(c);
    const p = parse('x + 1/2y');
    expect(p.ok ? null : p.error).toEqual({ code: 'ambiguous-division', at: 7, len: 1 });
  });
  it('error positions point into the typed text, not the normalised one', () => {
    // ⑴ becomes "(1)" under NFKC and 𝑥 is a surrogate pair: the mark must still land on what was typed
    const at = (s: string) => { const p = parse(s); return p.ok ? null : [p.error.at, p.error.len, s.slice(p.error.at, p.error.at + p.error.len)]; };
    expect(at('⑴ + foo')).toEqual([4, 3, 'foo']);
    expect(at('½x')).toEqual([0, 1, '½']); // ½ is 1⁄2, whose fraction slash is not division: the whole ½ is marked
    expect(at('𝑥 $ 1')).toEqual([3, 1, '$']);
    expect(at('ｘ ＋')).toEqual([2, 1, '＋']);
    expect(at('(x + 1')).toEqual([0, 1, '(']);
    expect(at('2 * sin(x')).toEqual([7, 1, '(']);
    expect(at('2 +  ')).toEqual([2, 1, '+']);
    const e = parse('   ');
    expect(e.ok ? null : [e.error.code, e.error.len]).toEqual(['empty', 0]);
  });
  it("a page's own variables: y is unknown on a page in x only", () => {
    expect(code('x + y', ['x'])).toBe('unknown-variable');
    expect(code('xy', ['x'])).toBe('unknown-name');
    const p = parse('t*x + 1', { vars: ['x', 't'] });
    expect(p.ok && p.vars).toEqual(['t', 'x']);
  });
  it('full-width input, real minus signs and superscripts', () => {
    expect(val('３ｘ＋１', { x: 2 })).toBe(7);
    expect(val('x⁻¹', { x: 4 })).toBe(0.25);
    expect(val('x² − 1', { x: 3 })).toBe(8);
    expect(val('（x）（x）', { x: 3 })).toBe(9);
  });
  it('length and depth limits', () => {
    expect(code('x+'.repeat(100) + 'x')).toBe('too-long');
    expect(code('('.repeat(61) + 'x' + ')'.repeat(61))).toBe('too-deep');
    expect(code('('.repeat(40) + 'x' + ')'.repeat(40))).toBe('ok');
  });
  it('no eval and no Function in the source', () => {
    const src = readFileSync(new URL('../src/lib/math/expr.ts', import.meta.url), 'utf8');
    expect(src).not.toMatch(/\beval\s*\(|\bFunction\s*\(|new\s+Function/);
  });
});

describe('domains: NaN from compile, a reason from diagnose', () => {
  it('real powers: odd roots of negatives are real, even roots NaN, 0 to a negative power NaN', () => {
    expect(val('(-8)^(1/3)')).toBeCloseTo(-2, 15);
    expect(val('(-8)^(2/3)')).toBeCloseTo(4, 14);
    expect(val('x^(1/3)', { x: -27 })).toBeCloseTo(-3, 14);
    expect(val('(-8)^0.5')).toBeNaN();
    expect(val('(-2)^3')).toBe(-8);
    expect(val('0^(-1)')).toBeNaN();
    expect(realPow(0, 0)).toBe(1);
  });
  it('compile returns NaN, diagnose names the rule', () => {
    const cases: [string, Record<string, number>, string][] = [
      ['1/x', { x: 0 }, 'division-by-zero'], ['sqrt(x)', { x: -1 }, 'sqrt-negative'], ['ln(x)', { x: 0 }, 'log-nonpositive'],
      ['log(x)', { x: -2 }, 'log-nonpositive'], ['asin(x)', { x: 1.5 }, 'asin-acos-range'], ['x^0.5', { x: -4 }, 'even-root-negative'],
      ['x^(-2)', { x: 0 }, 'pole'], ['tan(x)', { x: Math.PI / 2 }, 'pole'], ['exp(x)', { x: 1000 }, 'overflow'],
      ['x + y', { x: 1 }, 'missing-variable'],
    ];
    for (const [s, env, reason] of cases) {
      expect(diagnose(ast(s), env), s).toEqual({ domain: reason });
      if (reason !== 'overflow' && reason !== 'pole') expect(compile(ast(s))(env), s).toBeNaN();
    }
    expect(diagnose(ast('sqrt(x) + 1'), { x: 4 })).toEqual({ value: 3 });
  });
});

describe('printing', () => {
  const formulas = [
    '-x^2', '(-x)^2', 'x^2^3', '(x^2)^3', '2^-x', 'x - (y - t)', 'x/(y*t)', 'x/y*t', '-(x + y)', 'x*-y', '1/2x'.replace('2x', '(2x)'),
    'sin(x)^2 + cos(x)^2', 'e^(x*y)', 'abs(x - 1) + max(x, y)', '√(x + 1)', 'x²y', 'pow(x, t)', '-2^x', '(-2)^x',
  ];
  it('toText reads back to the same function', () => {
    const env = { x: 0.7, y: -1.3, t: 2.1 };
    for (const s of formulas) {
      const a = ast(s), back = ast(toText(a));
      const v1 = compile(a)(env), v2 = compile(back)(env);
      if (Number.isNaN(v1)) expect(v2, s).toBeNaN();
      else expect(v2, `${s} → ${toText(a)}`).toBe(v1);
      expect(toText(back)).toBe(toText(a)); // stable
    }
  });
  it('toTex', () => {
    expect(toTex(ast('x/(y + 1)'))).toBe('\\frac{x}{y + 1}');
    expect(toTex(ast('2x^2'))).toBe('2 x^{2}');
    expect(toTex(ast('x*2'))).toBe('x \\cdot 2');
    expect(toTex(ast('sin(x)^2'))).toBe('\\left(\\sin\\left(x\\right)\\right)^{2}');
    expect(toTex(ast('abs(x) + √x'))).toBe('\\left|x\\right| + \\sqrt{x}');
    expect(toTex(ast('log(x)'))).toBe('\\log_{10}\\left(x\\right)');
    expect(toTex(ast('-(x + 1)'))).toBe('-\\left(x + 1\\right)');
    expect(toTex(ast('x - (y - 1)'))).toBe('x - \\left(y - 1\\right)');
    expect(toTex(ast('x*-y'))).toBe('x \\left(-y\\right)');
  });
});

describe('symbolic derivatives', () => {
  const d = (s: string, v = 'x') => differentiate(ast(s), v);
  it('simplified closed forms', () => {
    expect(toText(d('x^2').ast)).toBe('2 * x');
    expect(toText(d('3x + 2').ast)).toBe('3');
    expect(toText(d('sin(x)').ast)).toBe('cos(x)');
    expect(toText(d('e^x').ast)).toBe('e^x');
    expect(toText(d('x*y', 'y').ast)).toBe('x');
    expect(toText(d('y^2 + t').ast)).toBe('0');
  });
  it('agree with a central difference everywhere they are defined', () => {
    const fs = [
      'x^3 - 2x', 'sin(x)cos(x)', 'tan(x)', 'asin(x/2)', 'acos(x/2)', 'atan(x^2)', 'sinh(x) - cosh(x)', 'tanh(x)',
      'exp(-x^2)', 'ln(x + 2)', 'log(x + 2)', 'sqrt(x + 2)', 'cbrt(x + 3)', 'x^x', '2^x', 'x/(1 + x^2)', 'pow(x + 2, 1.5)',
      'e^(sin(x))', '(x + 2)^(x + 1)', '1/(x - 5)', 'x^(1/3)',
    ];
    for (const s of fs) {
      const f = compile(ast(s)), df = compile(d(s).ast);
      for (const x of [-0.7, 0.4, 1.3]) {
        const h = 1e-5;
        const num = (f({ x: x + h }) - f({ x: x - h })) / (2 * h);
        if (!Number.isFinite(num)) continue;
        expect(Math.abs(df({ x }) - num), `${s} at ${x}`).toBeLessThan(1e-6 * Math.max(1, Math.abs(num)));
      }
    }
  });
  it('divergence and curl of a typed field are exact', () => {
    // F = (P, Q) = (x²y − sin y, x e^y + y³): div = 2xy + x e^y + 3y², curl = e^y − x² + cos y
    const P = ast('x^2*y - sin(y)'), Qf = ast('x e^y + y^3');
    const div = compile({ k: 'bin', op: '+', a: differentiate(P, 'x').ast, b: differentiate(Qf, 'y').ast });
    const curl = compile({ k: 'bin', op: '-', a: differentiate(Qf, 'x').ast, b: differentiate(P, 'y').ast });
    for (const [x, y] of [[0.3, -1.2], [2, 0.5], [-1.5, 2]]) {
      expect(div({ x, y })).toBeCloseTo(2 * x * y + x * Math.exp(y) + 3 * y * y, 13);
      expect(curl({ x, y })).toBeCloseTo(Math.exp(y) - x * x + Math.cos(y), 13);
    }
  });
  it('kinks and jumps are flagged as piecewise', () => {
    expect(d('abs(x)').piecewise).toBe(true);
    expect(compile(d('abs(x)').ast)({ x: -2 })).toBe(-1);
    expect(d('floor(x)').piecewise).toBe(true);
    expect(toText(d('floor(x)').ast)).toBe('0');
    const m = d('max(x, 2 - x)');
    expect(m.piecewise).toBe(true);
    expect(compile(m.ast)({ x: 3 })).toBe(1);
    expect(compile(m.ast)({ x: 0 })).toBe(-1);
    expect(compile(m.ast)({ x: 1 })).toBe(0); // a tie: the two slopes averaged
    expect(d('x^2').piecewise).toBe(false);
  });
  it('dependsOn', () => {
    expect(dependsOn(ast('x + sin(y)'), 'y')).toBe(true);
    expect(dependsOn(ast('x + sin(y)'), 't')).toBe(false);
  });
});

describe('Codex review 2026-10-07: regressions', () => {
  it('#16 negative bases: real only for exponents written as p/q with q odd, the same rule for the derivative', () => {
    expect(realPow(-1, 1000000000000.5)).toBeNaN();
    expect(val('(-1)^1000000000000.5')).toBeNaN();
    expect(val('x^2.0000000000015', { x: -2 })).toBeNaN();
    expect(compile(differentiate(ast('x^2.0000000000015'), 'x').ast)({ x: -2 })).toBeNaN();
    expect(val('x^0.3333333333333333', { x: -8 })).toBeNaN(); // a decimal is not the fraction 1/3
    expect(val('x^(1/3)', { x: -8 })).toBe(-2);
    expect(val('x^(-1/3)', { x: -8 })).toBe(-0.5);
    expect(val('pow(x, 2/3)', { x: -8 })).toBeCloseTo(4, 14);
    expect(diagnose(ast('x^(1/2)'), { x: -4 })).toEqual({ domain: 'even-root-negative' });
    // derivatives keep the fraction: (x^(1/3))′ = (1/3)x^(−2/3), (x^(2/3))′ = (2/3)x^(−1/3)
    const d1 = differentiate(ast('x^(1/3)'), 'x').ast, d2 = differentiate(ast('x^(2/3)'), 'x').ast;
    expect(toText(d1)).toBe('1 / 3 * x^(-2 / 3)');
    expect(compile(d1)({ x: -8 })).toBeCloseTo(1 / 12, 15);
    expect(compile(d2)({ x: -8 })).toBeCloseTo(-1 / 3, 15);
  });
  it('#17 toText → parse keeps every value bit for bit, long derivatives included', () => {
    const env = { x: 1, y: 1e308, t: 1e308 };
    for (const s of ['x - 1.0000000000000002', 'x*(y/t)', 'x + (y - t)', 'x/(y/t)', '1e-300*x', '0.1 + 0.2*x']) {
      const a = ast(s), back = ast(toText(a));
      const v1 = compile(a)(env), v2 = compile(back)(env);
      expect(Object.is(v1, v2), `${s} → ${toText(a)}: ${v1} vs ${v2}`).toBe(true);
    }
    expect(val('x - 1.0000000000000002', { x: 1 })).toBe(-(2 ** -52));
    // a derivative longer than the reader's 200-character limit reads back with the limits lifted
    const big = differentiate(ast('sin(x^2*y)^3 / (1 + exp(x - y)^2) + atan(x/y)^2 * ln(x^2 + y^2)'), 'x').ast;
    const text = toText(big);
    expect(text.length).toBeGreaterThan(200);
    const back = parse(text, { maxLength: Infinity, maxDepth: Infinity });
    expect(back.ok).toBe(true);
    if (back.ok) for (const [x, y] of [[0.7, 1.3], [-1.1, 0.4]]) expect(compile(back.ast)({ x, y })).toBe(compile(big)({ x, y }));
  });
});

describe('Codex re-review 2026-10-07: positions, literals, fraction limits', () => {
  const at = (s: string, opt?: Parameters<typeof parse>[1]) => {
    const p = parse(s, opt);
    return p.ok ? null : [p.error.code, p.error.at, p.error.len, s.slice(p.error.at, p.error.at + p.error.len)];
  };
  it('an astral character is marked whole (both halves of its surrogate pair), also past the length limit', () => {
    expect(at('x + 😀')).toEqual(['bad-character', 4, 2, '😀']);
    expect(at('😀😀😀', { maxLength: 2 })).toEqual(['too-long', 2, 2, '😀']); // the first character past the limit
    expect(at('x'.repeat(200) + '😀')).toEqual(['too-long', 200, 2, '😀']);
  });
  it('a number is marked as typed, not as printed', () => {
    expect(at('x 1e20 + y')).toEqual(['unexpected', 2, 4, '1e20']);
    expect(at('x 0.50 + y')).toEqual(['unexpected', 2, 4, '0.50']);
  });
  it('literals outside the doubles are refused', () => {
    expect(at('1e309')).toEqual(['bad-number', 0, 5, '1e309']);
    expect(at('2*1e-400')).toEqual(['bad-number', 2, 6, '1e-400']);
    expect(code('1e-310')).toBe('ok'); // subnormal, still a double
    expect(code('0e5')).toBe('ok');
  });
  it('invalid limit options are the program’s mistake: a RangeError, never at: undefined', () => {
    for (const bad of [0.5, 0, -1, NaN, 2.5]) {
      expect(() => parse('x', { maxLength: bad })).toThrow(RangeError);
      expect(() => parse('x', { maxDepth: bad })).toThrow(RangeError);
    }
    expect(parse('x', { maxLength: Infinity, maxDepth: 1 }).ok).toBe(true);
  });
  it('#16 exponent fractions are exact at any size, in value and derivative alike', () => {
    // −(2⁵³ − 1)/3: p odd, q odd → real for x = −1, and the derivative's exponent (p − 3)/3 stays exact
    const f = ast('x^(-9007199254740991/3)'), df = differentiate(f, 'x').ast;
    expect(compile(f)({ x: -1 })).toBe(-1);
    expect(compile(df)({ x: -1 })).toBe(-9007199254740991 / 3); // (p/q)·(−1)^((p − 3)/3), p − 3 even
    // −2⁵⁰/3: p even → +1 at −1; the derivative (p/q)·(−1)^((p − 3)/3) = 2⁵⁰/3 (Codex's value 375299968947541.3)
    const g = ast('x^(-1125899906842624/3)');
    expect(compile(g)({ x: -1 })).toBe(1);
    expect(compile(differentiate(g, 'x').ast)({ x: -1 })).toBeCloseTo(375299968947541.3, 1);
    expect(val('x^(1/3 + 1/3)', { x: -8 })).toBeCloseTo(4, 14); // 2/3, written as a sum
    expect(val('x^(1e20 - 1)', { x: -1 })).toBe(-1); // odd, although the double 1e20 − 1 is even
  });  it('#17 every literal parse accepts prints back to itself', () => {
    for (const s of ['1e308', '1e-310', '5e-324', '0.1', '123456789.123456789']) {
      const a = ast(s);
      expect(compile(ast(toText(a)))({})).toBe(compile(a)({}));
    }
  });
});

describe('poleFactors: where a formula has no value on a line it is defined on both sides of (D27)', () => {
  const factors = (src: string) => {
    const r = parse(src, { vars: ['x', 'y'] });
    if (!r.ok) throw new Error(src);
    return poleFactors(r.ast).map(toText);
  };
  it('denominators, negative constant powers, cos under tan; constants and one-sided domains are not poles', () => {
    expect(factors('-x/y')).toEqual(['y']);
    expect(factors('(y-1)/(x-0.1234567)')).toEqual(['x - 0.1234567']);
    expect(factors('x^(-1) + y^(-1/3)')).toEqual(['x', 'y']);
    expect(factors('pow(x + y, -2)')).toEqual(['x + y']);
    expect(factors('tan(x y)')).toEqual(['cos(x * y)']);
    expect(factors('1/(1/x)')).toEqual(['1 / x', 'x']);
    expect(factors('x/2 + sqrt(y) + ln(x)')).toEqual([]);
    expect(factors('3y^(2/3)')).toEqual([]);
  });
});

describe('exact values and smoothness at a point (E25: “=” only when exact; no derivative at a kink)', () => {
  const ast = (src: string) => { const r = parse(src, { vars: ['x', 'y'] }); if (!r.ok) throw new Error(src); return r.ast; };
  const at = (x: number | Q, y: number | Q) => ({ x: typeof x === 'number' ? Q.fromDouble(x) : x, y: typeof y === 'number' ? Q.fromDouble(y) : y });
  it('rational arithmetic, integer powers, special values', () => {
    expect(exactValue(ast('3x^2 + 3y^2'), at(1, 0))?.toString()).toBe('3');
    expect(exactValue(ast('cos(x) + cos(y)'), at(0, 0))?.toString()).toBe('2');
    expect(exactValue(ast('0.1x'), at(1, 0))?.toString()).toBe('1/10'); // the literal as typed, not its double
    expect(exactValue(ast('(y^2 - x^2)/(x^2 + y^2)^2'), at(2, 1))?.toString()).toBe('-3/25');
    expect(exactValue(ast('x^(-2)'), at(2, 0))?.toString()).toBe('1/4');
    expect(exactValue(ast('sqrt(x)'), at(Q.of(9, 4), Q.ZERO))?.toString()).toBe('3/2');
    expect(exactValue(ast('abs(x) + floor(y) + min(x, y)'), at(-1.5, 2.5))?.toString()).toBe('2');
  });
  it('numbers too large to print give up instead of building millions of digits (Codex review 2026-10-08)', () => {
    expect(exactValue(ast('(x^999)^999'), at(3, 0))).toBeNull();
    expect(exactValue(ast('((x^99)^99)^99'), at(1, 0))?.toString()).toBe('1');
    expect(exactValue(ast('(x^2)^3'), at(2, 0))?.toString()).toBe('64');
  });
  it('exactDerivative: forward mode over the rationals on the formula as typed (Codex review 2026-10-08)', () => {
    const d = (src: string, v: string, x: number | Q, y: number | Q) => exactDerivative(ast(src), v, at(x, y))?.d.toString() ?? null;
    expect(d('sin(x)', 'x', 0, 0)).toBe('1');
    expect(d('x/(x^2+y^2)', 'x', 2, 1)).toBe('-3/25');
    expect(d('x^3', 'x', Q.of(1, 2), 0)).toBe('3/4');
    expect(d('cos(1e-10)*x', 'x', 1, 0)).toBeNull(); // folded, cos(1e-10) is the double 1; it is not 1
    expect(d('0.99999999999999999*x', 'x', 1, 0)).toBe('99999999999999999/100000000000000000');
    expect(d('2.5e-3*x', 'x', 1, 0)).toBe('1/400');
    expect(d('abs(x - 1/3)', 'x', Q.of(1, 3), 0)).toBeNull(); // the kink itself
    expect(d('abs(x - 1/3)', 'x', 0, 0)).toBe('-1');
    expect(d('ln(y)', 'x', 1, 0)).toBeNull(); // ln 0: no value, so no derivative either
    expect(d('log(x)', 'x', 1, 0)).toBeNull(); // 1/ln 10
    expect(d('sqrt(x)', 'x', Q.of(9, 4), 0)).toBe('1/3');
    // nested pow is linear, not exponential, in the formula's size
    const nested = Array.from({ length: 24 }).reduce<string>((e) => `pow(${e}, 1)`, 'x');
    const t0 = performance.now();
    expect(exactValue(ast(`abs(${nested})`), at(0.5, 0))?.toString()).toBe('1/2');
    expect(smoothAt(ast(`abs(${nested})`), { x: 0.5, y: 0 })).toBe(true);
    expect(performance.now() - t0).toBeLessThan(500);
  });
  it('null where the value is not rational or not defined', () => {
    for (const src of ['cos(x)', 'pi x', 'sqrt(x)', 'e^x', 'x^(1/2)', '1/x']) expect(exactValue(ast(src), at(src === '1/x' ? 0 : 2, 0))).toBeNull();
    // the double says 1; the number is not
    expect(Math.cos(1e-10)).toBe(1);
    expect(exactValue(ast('cos(x)'), at(1e-10, 0))).toBeNull();
  });
  it('smoothAt: false exactly where abs, sign, floor, min or max switch', () => {
    expect(smoothAt(ast('abs(x)'), { x: 0, y: 0 })).toBe(false);
    expect(smoothAt(ast('abs(x)'), { x: 1e-300, y: 0 })).toBe(true);
    expect(smoothAt(ast('floor(y) + x'), { x: 0, y: 2 })).toBe(false);
    expect(smoothAt(ast('max(x, y)'), { x: 0.1, y: 0.1 })).toBe(false);
    expect(smoothAt(ast('sign(x - 0.1)'), { x: 0.1, y: 0 })).toBe(false); // 0.1 − 0.1 exactly, by the typed decimals
    expect(smoothAt(ast('sin(x) + y^3'), { x: 0, y: 0 })).toBe(true);
    // the derivative the library gives there (sign(0) = 0) is not a derivative: a page must not show it
    expect(compile(differentiate(ast('abs(x)'), 'x').ast)({ x: 0, y: 0 })).toBe(0);
  });
  it('optional rational inputs keep poles and branch boundaries at the original source point', () => {
    const env = { x: 1 / 49, y: 1 }, exact = { x: Q.of(1, 49), y: Q.ONE };
    expect(49 * env.x - 1).not.toBe(0); // rounded evaluation misses the boundary
    expect(diagnose(ast('1/(49*x-1)'), env, exact)).toEqual({ domain: 'division-by-zero' });
    expect(diagnose(ast('ln(49*x-1)'), env, exact)).toEqual({ domain: 'log-nonpositive' });
    for (const formula of ['abs(49*x-1)', 'sign(49*x-1)', 'floor(49*x)', 'min(49*x,1)', 'max(49*x,1)']) {
      expect(smoothAt(ast(formula), env, exact), formula).toBe(false);
    }
    expect(diagnose(ast('sin(y)+1/(49*x-1)'), env, exact)).toEqual({ domain: 'division-by-zero' });
    expect(diagnose(ast('sin(y)+49*x'), env, exact)).toEqual({ value: Math.sin(1) + 1 });
    expect(smoothAt(ast('sin(y)+49*x'), env, exact)).toBe(true);
    const above = { x: Q.ONE.add(Q.of(1, 1000000000000000000n)) };
    expect(diagnose(ast('asin(x)'), { x: 1 }, above)).toEqual({ domain: 'asin-acos-range' });
    // Optional precision does not change the formula grammar for negative bases and non-integer exponents.
    expect(diagnose(ast('(-8)^x'), { x: 1 / 3 }, { x: Q.of(1, 3) })).toEqual({ domain: 'even-root-negative' });
    expect(diagnose(ast('(-8)^(1/3)'), {}, {})).toEqual({ value: -2 });
    expect(diagnose(ast('1/(49*x-1)'), env)).toEqual({ value: 1 / (49 * env.x - 1) });
  });
});
