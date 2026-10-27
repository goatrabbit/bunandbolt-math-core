// A safe parser, compiler and differentiator for the formulas readers type (the formula input: y′ = f(x, y),
// F(x, y) = (P, Q), x′ = Ax …). No eval, no Function: the text becomes a small tree, the tree becomes closures.
//
// What is accepted
// - numbers: 2, 2.5, .5, 1e-3 (an "e" followed by digits is an exponent; elsewhere e is the constant);
// - variables (default x, y, t; a page passes its own set) and constants π / pi, e;
// - + − × ÷ ^ (also *, ·, /, **, the real minus sign), superscripts (x², x⁻¹), unary minus (−x² is −(x²)),
//   parentheses (or square brackets);
// - functions, always with parentheses: sin cos tan asin acos atan (arcsin … too) sinh cosh tanh exp ln log sqrt cbrt
//   abs sign floor, and min max pow with two arguments; √ before a number, a variable, a constant or a parenthesis;
// - implicit multiplication only where it cannot be misread: 2x, 2(x + 1), 2 sin(x), x(y + 1), (x)(y), xy (a run of
//   single-letter variables), πx, x²y, e^(2x)y. Refused with a code asking for parentheses or ×: after a division
//   ("1/2x"), after a bare exponent ("e^2x", "x^2 y"), and before a number ("x2").
// - log is base 10, ln is natural (ISO 80000-2 writes lg for base 10; pages say which in their hint).
// - a negative number to a non-integer power is real only for an exponent written as a fraction p/q with q odd:
//   (−8)^(1/3) = −2, x^(2/3) for negative x; (−8)^0.5, x^0.3333 and x^t for a variable t are NaN there (a page wanting
//   the real cube root offers cbrt). The rule looks at the formula, not at how close a double is to some p/q: an
//   exponent written as an exact rational of integer literals (2/3, −1/3, and sums and differences of them: 1 + 1/3) is
//   read exactly, at any size, so x^(1/3 + 1/3) is real for negative x just as x^(2/3) is; its derivative's exponent
//   p/q − 1 stays exact too.
// - number literals must be doubles: 1e309 (beyond the range) and 1e-400 (would round to 0) are refused as bad-number.
// Text is normalised with NFKC first (superscripts excepted), so full-width digits, letters and brackets work.
//
// Errors are codes with a position (the page turns them into en / zh text). Evaluation never throws: outside a
// function's domain the compiled function returns NaN, and `diagnose` says which domain rule failed.
// diagnose/smoothAt optionally accept original rational inputs: supported rational subexpressions determine domain
// and branch boundaries exactly; transcendental values remain numerical. Existing two-argument calls are unchanged.

import { parseExact, Q } from './rational';

export type Fn =
  | 'sin' | 'cos' | 'tan' | 'asin' | 'acos' | 'atan' | 'sinh' | 'cosh' | 'tanh'
  | 'exp' | 'ln' | 'log' | 'sqrt' | 'cbrt' | 'abs' | 'sign' | 'floor' | 'min' | 'max' | 'pow';

export type Node =
  /** `src`: the literal as typed (from `parse`; nodes made by `simplify` have none), for exact readings of it */
  | { k: 'num'; v: number; src?: string }
  | { k: 'var'; name: string }
  | { k: 'const'; name: 'pi' | 'e' }
  | { k: 'neg'; a: Node }
  | { k: 'bin'; op: '+' | '-' | '*' | '/' | '^'; a: Node; b: Node }
  | { k: 'call'; fn: Fn; args: Node[] };

export type ErrorCode =
  | 'empty' // nothing typed
  | 'too-long' // more than 200 characters
  | 'too-deep' // nested more than 60 levels
  | 'bad-character' // a character that means nothing here
  | 'bad-number' // "1.2.3", "2e-"
  | 'unknown-name' // a word that is not a variable, constant or function
  | 'unknown-variable' // a variable this page does not have (e.g. y on a page in x only)
  | 'function-needs-paren' // "sin x", "sinx": functions need parentheses
  | 'wrong-arity' // min(x), sin(x, y)
  | 'missing-operand' // "2 +", "* x"
  | 'missing-paren' // "(x + 1"
  | 'extra-paren' // "x + 1)"
  | 'unexpected' // a token where it cannot go ("x 2", "1, 2")
  | 'ambiguous-division' // "1/2x": write (1/2)x or 1/(2x)
  | 'ambiguous-power' // "e^2x", "√x^2": write e^(2x) or (√x)^2
  | 'empty-parens'; // "()"

export interface ParseError {
  code: ErrorCode;
  /** where the problem starts, as an index into the text the reader typed (UTF-16, so it can mark the characters) */
  at: number;
  /** how many UTF-16 units of the typed text to mark — always whole typed characters (an emoji is 2) — at least 1; 0
   *  only for "nothing typed" */
  len: number;
  /** the offending word or symbol, when there is one (normalised) */
  detail?: string;
}

export type Parsed = { ok: true; ast: Node; vars: string[] } | { ok: false; error: ParseError };

const FUNCS: Record<string, Fn> = {
  sin: 'sin', cos: 'cos', tan: 'tan', asin: 'asin', acos: 'acos', atan: 'atan', arcsin: 'asin', arccos: 'acos', arctan: 'atan',
  sinh: 'sinh', cosh: 'cosh', tanh: 'tanh', exp: 'exp', ln: 'ln', log: 'log', sqrt: 'sqrt', cbrt: 'cbrt', abs: 'abs',
  sign: 'sign', sgn: 'sign', floor: 'floor', min: 'min', max: 'max', pow: 'pow',
};
const ARITY: Record<Fn, number> = {
  sin: 1, cos: 1, tan: 1, asin: 1, acos: 1, atan: 1, sinh: 1, cosh: 1, tanh: 1, exp: 1, ln: 1, log: 1, sqrt: 1, cbrt: 1,
  abs: 1, sign: 1, floor: 1, min: 2, max: 2, pow: 2,
};
const SUP: Record<string, string> = { '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-' };
const OPS: Record<string, '+' | '-' | '*' | '/' | '^'> = { '+': '+', '-': '-', '−': '-', '–': '-', '*': '*', '×': '*', '·': '*', '⋅': '*', '/': '/', '÷': '/', '^': '^' };

/** Own keys only: a plain lookup table also answers "constructor", "toString", "__proto__" … from its prototype. */
const has = (table: object, key: string) => Object.prototype.hasOwnProperty.call(table, key);

type Tok =
  | { t: 'num'; v: number; at: number; text: string }
  | { t: 'var'; name: string; at: number }
  | { t: 'const'; name: 'pi' | 'e'; at: number }
  | { t: 'fn'; fn: Fn; at: number; text: string }
  | { t: 'op'; op: '+' | '-' | '*' | '/' | '^'; at: number }
  | { t: 'sup'; v: number; at: number }
  | { t: 'root'; at: number }
  | { t: '('; at: number }
  | { t: ')'; at: number }
  | { t: ','; at: number };

class Fail extends Error {
  constructor(readonly error: ParseError) {
    super(error.code);
  }
}
const fail = (code: ErrorCode, at: number, detail?: string): never => {
  throw new Fail({ code, at, len: 0, detail });
};

function tokenize(s: string, vars: Set<string>): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === ' ' || ch === '\t') {
      i++;
      continue;
    }
    if (/[0-9.]/.test(ch)) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(s.slice(i));
      if (!m) return fail('bad-number', i, ch);
      const text = m[0], rest = s.slice(i + text.length);
      // "1.2.3", or an exponent sign with nothing after it ("2e-")
      if (rest[0] === '.' || /^[eE][+-](?!\d)/.test(rest)) fail('bad-number', i, text + rest.slice(0, 2));
      const v = Number(text);
      // a literal beyond the double range (1e309), or one that underflows to 0 although it has digits (1e-400)
      if (!Number.isFinite(v) || (v === 0 && /[1-9]/.test(text.replace(/[eE].*$/, '')))) fail('bad-number', i, text);
      out.push({ t: 'num', v, at: i, text });
      i += text.length;
      continue;
    }
    if (has(SUP, ch)) {
      const run = /^[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+/.exec(s.slice(i))![0];
      const digits = [...run].map((c) => SUP[c]).join('');
      if (!/^-?\d+$/.test(digits)) fail('bad-number', i, run);
      out.push({ t: 'sup', v: Number(digits), at: i });
      i += run.length;
      continue;
    }
    if (ch === 'π') {
      out.push({ t: 'const', name: 'pi', at: i });
      i++;
      continue;
    }
    if (/[a-zA-Z]/.test(ch)) {
      const word = /^[a-zA-Z]+/.exec(s.slice(i))![0];
      const lower = word.toLowerCase();
      if (has(FUNCS, lower)) out.push({ t: 'fn', fn: FUNCS[lower], at: i, text: word });
      else if (lower === 'pi') out.push({ t: 'const', name: 'pi', at: i });
      else if (word.length === 1 && vars.has(word)) out.push({ t: 'var', name: word, at: i });
      else if (word === 'e') out.push({ t: 'const', name: 'e', at: i });
      else if ([...word].every((c) => vars.has(c) || c === 'e')) {
        // a run of single-letter variables (and e): xy → x·y
        [...word].forEach((c, k) => out.push(vars.has(c) ? { t: 'var', name: c, at: i + k } : { t: 'const', name: 'e', at: i + k }));
      } else {
        // "sinx", "lnx": a function name run into its argument
        const head = Object.keys(FUNCS).filter((f) => lower.startsWith(f) && [...word.slice(f.length)].every((c) => vars.has(c)));
        if (head.length) fail('function-needs-paren', i, word.slice(0, head.sort((a, b) => b.length - a.length)[0].length));
        fail(word.length === 1 ? 'unknown-variable' : 'unknown-name', i, word);
      }
      i += word.length;
      continue;
    }
    if (ch === '*' && s[i + 1] === '*') {
      out.push({ t: 'op', op: '^', at: i });
      i += 2;
      continue;
    }
    if (has(OPS, ch)) {
      out.push({ t: 'op', op: OPS[ch], at: i });
      i++;
      continue;
    }
    if (ch === '√') {
      out.push({ t: 'root', at: i });
      i++;
      continue;
    }
    if (ch === '(' || ch === '[') {
      out.push({ t: '(', at: i });
      i++;
      continue;
    }
    if (ch === ')' || ch === ']') {
      out.push({ t: ')', at: i });
      i++;
      continue;
    }
    if (ch === ',') {
      out.push({ t: ',', at: i });
      i++;
      continue;
    }
    fail('bad-character', i, String.fromCodePoint(s.codePointAt(i)!)); // the whole character, not half a surrogate pair
  }
  return out;
}

const MAX_LEN = 200, MAX_DEPTH = 60;

/** NFKC per character, leaving superscript digits alone (NFKC would turn x² into x2). */
const normalise = (s: string) => [...s].map((c) => (has(SUP, c) ? c : c.normalize('NFKC'))).join('');
/**
 * For each UTF-16 index of the normalised text (and one past its end), where the typed character it came from starts
 * and ends in the typed text: a character can grow under NFKC (½ → 1⁄2) or be a surrogate pair (😀), so an error is
 * mapped back and always marks whole typed characters.
 */
function rawSpans(input: string): { start: number[]; end: number[] } {
  const start: number[] = [], end: number[] = [];
  let raw = 0;
  for (const c of input) {
    const n = (has(SUP, c) ? c : c.normalize('NFKC')).length;
    for (let i = 0; i < n; i++) {
      start.push(raw);
      end.push(raw + c.length);
    }
    raw += c.length;
  }
  start.push(raw);
  end.push(raw);
  return { start, end };
}

/** A limit option: a whole number ≥ 1, or Infinity. */
const limitOk = (v: number) => v === Infinity || (Number.isInteger(v) && v >= 1);

/**
 * Parse a typed formula. `vars` are the variables this page has (default x, y, t); `maxLength` (200) and `maxDepth`
 * (60) guard what readers type and can be lifted for text the program generated itself.
 */
export function parse(input: string, opt: { vars?: string[]; maxLength?: number; maxDepth?: number } = {}): Parsed {
  // a bad limit is the program's mistake, not the reader's: refuse it loudly
  if (opt.maxLength !== undefined && !limitOk(opt.maxLength)) throw new RangeError('parse: maxLength must be a whole number ≥ 1 or Infinity');
  if (opt.maxDepth !== undefined && !limitOk(opt.maxDepth)) throw new RangeError('parse: maxDepth must be a whole number ≥ 1 or Infinity');
  const text = normalise(input);
  const varSet = new Set(opt.vars ?? ['x', 'y', 't']);
  try {
    if (!text.trim()) fail('empty', 0);
    const maxLength = opt.maxLength ?? MAX_LEN;
    if (text.length > maxLength) fail('too-long', maxLength);
    const toks = tokenize(text, varSet);
    // matching parentheses, so "the exponent was exactly one parenthesised group" can be checked
    const match = new Map<number, number>();
    const open: number[] = [];
    toks.forEach((tk, i) => {
      if (tk.t === '(') open.push(i);
      else if (tk.t === ')' && open.length) match.set(open.pop()!, i);
    });
    let pos = 0, depth = 0;
    const peek = () => toks[pos];
    const isOp = (tk: Tok | undefined, ...ops: string[]) => tk?.t === 'op' && ops.includes(tk.op);
    const startsOperand = (tk: Tok | undefined) => !!tk && (tk.t === 'num' || tk.t === 'var' || tk.t === 'const' || tk.t === 'fn' || tk.t === '(' || tk.t === 'root');
    // an implicit product may begin with these (never with a number)
    const canImplicit = (tk: Tok | undefined) => !!tk && (tk.t === 'var' || tk.t === 'const' || tk.t === 'fn' || tk.t === '(' || tk.t === 'root');
    const needOperand = (after: Tok, signs: boolean) => {
      const nx = peek();
      if (!startsOperand(nx) && !(signs && isOp(nx, '+', '-'))) fail('missing-operand', nx ? nx.at : after.at + 1);
    };

    function expr(): Node {
      if (++depth > (opt.maxDepth ?? MAX_DEPTH)) fail('too-deep', peek()?.at ?? text.length);
      let a = term();
      for (let tk = peek(); tk && tk.t === 'op' && (tk.op === '+' || tk.op === '-'); tk = peek()) {
        pos++;
        needOperand(tk, true);
        a = { k: 'bin', op: tk.op, a, b: term() };
      }
      depth--;
      return a;
    }

    function term(): Node {
      let a = unary();
      let afterDivision = false;
      for (;;) {
        const tk = peek();
        if (tk && tk.t === 'op' && (tk.op === '*' || tk.op === '/')) {
          pos++;
          needOperand(tk, true);
          a = { k: 'bin', op: tk.op, a, b: unary() };
          afterDivision = tk.op === '/';
          continue;
        }
        if (tk?.t === 'num') fail('unexpected', tk.at, tk.text); // the number as typed, not as printed (1e20)
        if (canImplicit(tk)) {
          if (afterDivision) fail('ambiguous-division', tk!.at);
          a = { k: 'bin', op: '*', a, b: unary(true) };
          continue;
        }
        return a;
      }
    }

    // leading signs; `implicit` = the right side of an implicit product, which takes no sign
    function unary(implicit = false): Node {
      const tk = peek();
      if (!implicit && tk && tk.t === 'op' && (tk.op === '-' || tk.op === '+')) {
        pos++;
        needOperand(tk, true);
        const a = unary();
        return tk.op === '-' ? { k: 'neg', a } : a;
      }
      return power();
    }

    function power(): Node {
      let base = atom();
      const sup = peek();
      if (sup?.t === 'sup') {
        pos++;
        base = { k: 'bin', op: '^', a: base, b: sup.v < 0 ? { k: 'neg', a: { k: 'num', v: -sup.v } } : { k: 'num', v: sup.v } };
        if (isOp(peek(), '^') || peek()?.t === 'sup') fail('ambiguous-power', peek()!.at);
        return base;
      }
      const tk = peek();
      if (tk && tk.t === 'op' && tk.op === '^') {
        pos++;
        needOperand(tk, true);
        const start = pos;
        const exp = unary(); // right-associative; allows 2^-x
        // "e^2x": a factor right after a bare exponent could belong to it; after "e^(2x)" it cannot
        const grouped = toks[start].t === '(' && match.get(start) === pos - 1;
        if (!grouped && (canImplicit(peek()) || peek()?.t === 'num')) fail('ambiguous-power', peek()!.at);
        return { k: 'bin', op: '^', a: base, b: exp };
      }
      return base;
    }

    function atom(): Node {
      const tk = peek();
      if (!tk) return fail('missing-operand', text.length);
      pos++;
      switch (tk.t) {
        case 'num':
          return { k: 'num', v: tk.v, src: tk.text };
        case 'var':
          return { k: 'var', name: tk.name };
        case 'const':
          return { k: 'const', name: tk.name };
        case 'root': {
          const nx = peek();
          if (!nx || !(nx.t === 'num' || nx.t === 'var' || nx.t === 'const' || nx.t === '(')) return fail('missing-operand', nx ? nx.at : tk.at + 1);
          const arg = atom();
          if (isOp(peek(), '^') || peek()?.t === 'sup') fail('ambiguous-power', peek()!.at);
          return { k: 'call', fn: 'sqrt', args: [arg] };
        }
        case 'fn': {
          if (peek()?.t !== '(') return fail('function-needs-paren', tk.at, tk.text);
          const openAt = peek()!.at; // a parenthesis never closed is marked where it opened
          pos++;
          if (peek()?.t === ')') fail('empty-parens', peek()!.at);
          const args: Node[] = [];
          for (;;) {
            args.push(expr());
            const nx = peek();
            if (nx?.t === ',') {
              pos++;
              continue;
            }
            if (nx?.t === ')') {
              pos++;
              break;
            }
            return fail(nx ? 'unexpected' : 'missing-paren', nx ? nx.at : openAt);
          }
          if (args.length !== ARITY[tk.fn]) fail('wrong-arity', tk.at, tk.text);
          return { k: 'call', fn: tk.fn, args };
        }
        case '(': {
          if (peek()?.t === ')') fail('empty-parens', peek()!.at);
          const inner = expr();
          const close = peek();
          if (!close || close.t !== ')') return fail(close ? 'unexpected' : 'missing-paren', close ? close.at : tk.at);
          pos++;
          return inner;
        }
        case ')':
          return fail('extra-paren', tk.at);
        case 'op':
          return fail('missing-operand', tk.at, tk.op);
        default:
          return fail('unexpected', tk.at);
      }
    }

    const ast = expr();
    if (pos < toks.length) {
      const tk = toks[pos];
      fail(tk.t === ')' ? 'extra-paren' : 'unexpected', tk.at);
    }
    const used = new Set<string>();
    walk(ast, (n) => n.k === 'var' && used.add(n.name));
    return { ok: true, ast, vars: [...used].sort() };
  } catch (e) {
    if (e instanceof Fail) {
      // positions come out in the normalised text; the reader's marks belong in what they typed
      const { start, end } = rawSpans(input);
      const { code, at, detail } = e.error;
      const a = Math.min(Math.max(0, at), start.length - 1);
      const last = Math.min(a + Math.max(1, detail?.length ?? 1), start.length - 1) - 1; // last marked normalised unit
      let rawAt = start[a], rawEnd = Math.max(last >= a ? end[last] : rawAt, rawAt + 1);
      // a problem found at the very end ("2 +", "(x + 1" with nothing after) marks the last thing typed, so there is
      // always a character to underline
      const typed = input.trimEnd();
      if (code !== 'empty' && rawAt >= typed.length && typed.length) {
        const last = [...typed].pop()!;
        rawAt = typed.length - last.length;
        rawEnd = typed.length;
      }
      return { ok: false, error: { code, at: rawAt, len: code === 'empty' ? 0 : Math.min(input.length, rawEnd) - rawAt, ...(detail !== undefined ? { detail } : {}) } };
    }
    throw e;
  }
}

function walk(n: Node, f: (n: Node) => void) {
  f(n);
  if (n.k === 'neg') walk(n.a, f);
  else if (n.k === 'bin') {
    walk(n.a, f);
    walk(n.b, f);
  } else if (n.k === 'call') n.args.forEach((a) => walk(a, f));
}

/** Does the expression depend on variable v? */
export function dependsOn(n: Node, v: string): boolean {
  let hit = false;
  walk(n, (m) => {
    if (m.k === 'var' && m.name === v) hit = true;
  });
  return hit;
}

/**
 * The expressions whose zeros are where the formula has a pole — where it has no value on a line it is defined on
 * both sides of (D27, the vector-field family): every denominator, the base of a negative constant power (x⁻¹,
 * x^(−1/3)), and cos u under tan u. A step of a numerical method can jump such a line unseen (the formula is NaN only
 * on the line itself): a page passes them as a barrier whose sign change stops the step. Constants are left out (a
 * zero constant denominator makes the whole formula NaN, which a solver sees). Boundaries of a domain with no value on
 * one side (√u, ln u for u < 0) are not poles: the formula is NaN beyond them, which a solver also sees.
 */
export function poleFactors(n: Node): Node[] {
  const out: Node[] = [];
  const varying = (m: Node) => { let v = false; walk(m, (q) => { if (q.k === 'var') v = true; }); return v; };
  const negConst = (e: Node) => !varying(e) && compile(e)({}) < 0; // a constant exponent below 0 (−1, −1/3, −(2 + 1))
  walk(n, (m) => {
    if (m.k === 'bin' && m.op === '/' && varying(m.b)) out.push(m.b);
    if (m.k === 'bin' && m.op === '^' && negConst(m.b) && varying(m.a)) out.push(m.a);
    if (m.k === 'call' && m.fn === 'pow' && negConst(m.args[1]) && varying(m.args[0])) out.push(m.args[0]);
    if (m.k === 'call' && m.fn === 'tan' && varying(m.args[0])) out.push({ k: 'call', fn: 'cos', args: [m.args[0]] });
  });
  return out;
}

// ---------------------------------------------------------------- evaluation

export type Env = Record<string, number>;
export type Compiled = (env: Env) => number;

/**
 * Real power for a numeric exponent: a negative base needs an integer exponent (else NaN); 0 to a negative power is NaN
 * (a pole), as division by zero is. Exponents written as fractions go through `ratPow`.
 */
export function realPow(base: number, exp: number): number {
  if (base === 0 && exp < 0) return NaN;
  if (base < 0 && !Number.isInteger(exp)) return NaN;
  return Math.pow(base, exp);
}

/** base^(p/q) for integers p, q > 0: a negative base has the real value when q is odd (−8^(1/3) = −2), else NaN. */
function ratPow(base: number, p: bigint, q: bigint): number {
  if (base === 0 && p < 0n) return NaN;
  if (base >= 0) return Math.pow(base, Number(p) / Number(q));
  if (q % 2n === 0n) return NaN;
  const mag = Math.pow(-base, Number(p) / Number(q));
  return p % 2n === 0n ? mag : -mag;
}

/**
 * The exponent as an exact fraction p/q (BigInt, q > 0, lowest terms) when the formula writes it as an exact rational:
 * integer literals combined with negation, /, + and − (2/3, −1/3, 1 + 1/3, and the derivative's b − 1), at any size.
 * Anything else (a decimal, π, a variable) is null.
 */
function exactFraction(n: Node): { p: bigint; q: bigint } | null {
  if (n.k === 'num') return Number.isInteger(n.v) ? { p: BigInt(n.v), q: 1n } : null;
  if (n.k === 'neg') {
    const f = exactFraction(n.a);
    return f && { p: -f.p, q: f.q };
  }
  if (n.k === 'bin' && (n.op === '/' || n.op === '+' || n.op === '-')) {
    const a = exactFraction(n.a), b = exactFraction(n.b);
    if (!a || !b) return null;
    let p: bigint, q: bigint;
    if (n.op === '/') {
      if (b.p === 0n) return null;
      p = a.p * b.q;
      q = a.q * b.p;
    } else {
      p = n.op === '+' ? a.p * b.q + b.p * a.q : a.p * b.q - b.p * a.q;
      q = a.q * b.q;
    }
    if (q < 0n) [p, q] = [-p, -q];
    const g = gcdBig(p < 0n ? -p : p, q);
    return { p: p / g, q: q / g };
  }
  return null;
}
const gcdBig = (a: bigint, b: bigint): bigint => (b === 0n ? a || 1n : gcdBig(b, a % b));
const SAFE = BigInt(Number.MAX_SAFE_INTEGER);
/** An exact fraction as a tree (an integer, or p/q), when p and q are safe integers; else null. */
function fractionNode(f: { p: bigint; q: bigint }): Node | null {
  if (f.p > SAFE || -f.p > SAFE || f.q > SAFE) return null;
  return f.q === 1n ? num(Number(f.p)) : { k: 'bin', op: '/', a: num(Number(f.p)), b: num(Number(f.q)) };
}

/** The power base^exponent for a compiled base, choosing the fraction rule when the exponent is written as one. */
function powerOf(exponent: Node): ((base: number, exp: number) => number) {
  const f = exactFraction(exponent);
  // exact exponents (integers included) take their sign from the exact parity of p, not from a rounded double
  return f ? (base) => ratPow(base, f.p, f.q) : realPow;
}

const apply1: Record<Exclude<Fn, 'min' | 'max' | 'pow'>, (u: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, exp: Math.exp, ln: (u) => (u > 0 ? Math.log(u) : NaN),
  log: (u) => (u > 0 ? Math.log10(u) : NaN), sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, sign: Math.sign, floor: Math.floor,
};

/**
 * Compile to a closure (env → value). Never throws: NaN wherever `diagnose` reports a domain problem (except tan, which
 * returns the huge finite value rounding gives near its poles), ±Infinity only on overflow.
 */
export function compile(n: Node): Compiled {
  switch (n.k) {
    case 'num': {
      const v = n.v;
      return () => v;
    }
    case 'const': {
      const v = n.name === 'pi' ? Math.PI : Math.E;
      return () => v;
    }
    case 'var': {
      const name = n.name;
      return (env) => (has(env, name) ? env[name] : NaN);
    }
    case 'neg': {
      const a = compile(n.a);
      return (env) => -a(env);
    }
    case 'bin': {
      const a = compile(n.a), b = compile(n.b);
      switch (n.op) {
        case '+': return (env) => a(env) + b(env);
        case '-': return (env) => a(env) - b(env);
        case '*': return (env) => a(env) * b(env);
        case '/': return (env) => {
          const d = b(env);
          return d === 0 ? NaN : a(env) / d;
        };
        case '^': {
          const pw = powerOf(n.b);
          return (env) => pw(a(env), b(env));
        }
      }
      break;
    }
    case 'call': {
      const args = n.args.map(compile);
      if (n.fn === 'min') return (env) => Math.min(args[0](env), args[1](env));
      if (n.fn === 'max') return (env) => Math.max(args[0](env), args[1](env));
      if (n.fn === 'pow') {
        const pw = powerOf(n.args[1]);
        return (env) => pw(args[0](env), args[1](env));
      }
      const f = apply1[n.fn], u = args[0];
      return (env) => f(u(env));
    }
  }
  throw new Error('unreachable');
}

export type DomainCode =
  | 'division-by-zero'
  | 'sqrt-negative'
  | 'log-nonpositive'
  | 'asin-acos-range'
  | 'even-root-negative' // a negative base to a power that is not p/q with q odd
  | 'pole' // 0 to a negative power; tan where cos is 0 to rounding
  | 'overflow'
  | 'missing-variable';

/** Evaluate with checks: the value, or the first domain rule that failed (for the page's message).
 *  An optional exact environment keeps rational subexpressions and domain comparisons at the supplied source
 *  point, before rounding their values for non-rational functions. It does not make transcendental evaluation
 *  exact; unsupported exact subexpressions retain the numerical checks. Without it, behavior is unchanged. */
export function diagnose(n: Node, env: Env, exactEnv?: Record<string, Q>): { value: number } | { domain: DomainCode } {
  try {
    const v = evalChecked(n, env, exactEnv);
    return Number.isFinite(v) ? { value: v } : { domain: 'overflow' };
  } catch (e) {
    if (e instanceof DomainError) return { domain: e.code };
    throw e;
  }
}

class DomainError extends Error {
  constructor(readonly code: DomainCode) {
    super(code);
  }
}
const domain = (code: DomainCode): never => {
  throw new DomainError(code);
};

function checkedPow(a: number, b: number, exponent: Node, exactBase?: Q | null, exactExponent?: Q | null): number {
  if ((exactBase ? exactBase.isZero() : a === 0) && (exactExponent ? exactExponent.sign() < 0 : b < 0)) return domain('pole');
  if (exactBase?.sign() === -1) {
    const fraction = exactFraction(exponent);
    if (fraction ? fraction.q % 2n === 0n : exactExponent ? !exactExponent.isInteger() : !Number.isInteger(b)) return domain('even-root-negative');
  }
  const v = powerOf(exponent)(a, b);
  return Number.isNaN(v) && !Number.isNaN(a) && !Number.isNaN(b) ? domain('even-root-negative') : v;
}

function evalChecked(n: Node, env: Env, exactEnv?: Record<string, Q>): number {
  const exact = (m: Node) => exactEnv ? exactValue(m, exactEnv) : null;
  const whole = exact(n);
  if (whole) return whole.toNumber();
  const ev = (m: Node) => evalChecked(m, env, exactEnv);
  switch (n.k) {
    case 'num': return n.v;
    case 'const': return n.name === 'pi' ? Math.PI : Math.E;
    case 'var': return has(env, n.name) ? env[n.name] : domain('missing-variable');
    case 'neg': return -ev(n.a);
    case 'bin': {
      const a = ev(n.a), b = ev(n.b);
      switch (n.op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return (exact(n.b)?.isZero() ?? b === 0) ? domain('division-by-zero') : a / b;
        case '^': return checkedPow(a, b, n.b, exact(n.a), exact(n.b));
      }
      break;
    }
    case 'call': {
      const u = ev(n.args[0]), q = exact(n.args[0]);
      switch (n.fn) {
        case 'sqrt': return (q ? q.sign() < 0 : u < 0) ? domain('sqrt-negative') : Math.sqrt(u);
        case 'ln':
        case 'log': return (q ? q.sign() <= 0 : u <= 0) ? domain('log-nonpositive') : n.fn === 'ln' ? Math.log(u) : Math.log10(u);
        case 'asin':
        case 'acos': return (q ? q.abs().cmp(Q.ONE) > 0 : Math.abs(u) > 1) ? domain('asin-acos-range') : n.fn === 'asin' ? Math.asin(u) : Math.acos(u);
        case 'tan': return Math.abs(Math.cos(u)) < 1e-15 ? domain('pole') : Math.tan(u);
        case 'min': return Math.min(u, ev(n.args[1]));
        case 'max': return Math.max(u, ev(n.args[1]));
        case 'pow': return checkedPow(u, ev(n.args[1]), n.args[1], q, exact(n.args[1]));
        default: return apply1[n.fn](u);
      }
    }
  }
  throw new Error('unreachable');
}

// ---------------------------------------------------------------- simplification and differentiation

const num = (v: number): Node => ({ k: 'num', v });
const isNum = (n: Node, v?: number) => n.k === 'num' && (v === undefined || n.v === v);

/** The value of a variable-free subtree, or null if it is undefined or not finite. */
function fold(n: Node): number | null {
  const r = diagnose(n, {});
  return 'value' in r ? r.value : null;
}

/**
 * Fold constants and drop 0 / 1 identities. Equal to the input wherever the input is defined; dropping 0·u or 0/u
 * may define it where u was not (fine for derivatives, which are read where the formula is defined).
 */
export function simplify(n: Node): Node {
  switch (n.k) {
    case 'num':
    case 'var':
    case 'const':
      return n;
    case 'neg': {
      const a = simplify(n.a);
      if (a.k === 'num') return num(-a.v);
      if (a.k === 'neg') return a.a;
      return { k: 'neg', a };
    }
    case 'bin': {
      const a = simplify(n.a), b = simplify(n.b);
      // exact rationals of integer literals fold exactly, to an integer or a fraction in lowest terms (exponents keep
      // their meaning); when the result is beyond the safe integers they stay as written (exactFraction still reads them)
      const f = n.op === '/' || n.op === '+' || n.op === '-' ? exactFraction({ k: 'bin', op: n.op, a, b }) : null;
      if (f) {
        const node = fractionNode(f);
        if (node) return node;
        return { k: 'bin', op: n.op, a, b };
      }
      if (a.k === 'num' && b.k === 'num') {
        const v = fold({ k: 'bin', op: n.op, a, b });
        if (v !== null) return num(v);
      }
      switch (n.op) {
        case '+':
          if (isNum(a, 0)) return b;
          if (isNum(b, 0)) return a;
          if (b.k === 'neg') return simplify({ k: 'bin', op: '-', a, b: b.a });
          break;
        case '-':
          if (isNum(b, 0)) return a;
          if (isNum(a, 0)) return simplify({ k: 'neg', a: b });
          if (b.k === 'neg') return simplify({ k: 'bin', op: '+', a, b: b.a });
          break;
        case '*':
          if (isNum(a, 0) || isNum(b, 0)) return num(0);
          if (isNum(a, 1)) return b;
          if (isNum(b, 1)) return a;
          if (isNum(a, -1)) return simplify({ k: 'neg', a: b });
          if (isNum(b, -1)) return simplify({ k: 'neg', a });
          break;
        case '/':
          if (isNum(a, 0) && !isNum(b, 0)) return num(0);
          if (isNum(b, 1)) return a;
          break;
        case '^':
          if (isNum(b, 0)) return num(1);
          if (isNum(b, 1)) return a;
          break;
      }
      return { k: 'bin', op: n.op, a, b };
    }
    case 'call': {
      const args = n.args.map(simplify);
      if (args.every((x) => x.k === 'num')) {
        const v = fold({ k: 'call', fn: n.fn, args });
        if (v !== null) return num(v);
      }
      return { k: 'call', fn: n.fn, args };
    }
  }
}

const B = (op: '+' | '-' | '*' | '/' | '^', a: Node, b: Node): Node => ({ k: 'bin', op, a, b });
const C = (fn: Fn, ...args: Node[]): Node => ({ k: 'call', fn, args });

export interface Derivative {
  ast: Node;
  /** true when the formula uses abs, sign, floor, min or max: the derivative holds except where those bend or jump */
  piecewise: boolean;
}

/**
 * Symbolic ∂/∂v, simplified: exact wherever the formula is differentiable (so divergence and curl of a typed field are
 * exact, not differenced). abs′ = sign; sign′ = floor′ = 0; min / max take the active argument's slope, the average of
 * the two at a tie.
 */
export function differentiate(n: Node, v: string): Derivative {
  let piecewise = false;
  function d(n: Node): Node {
    if (!dependsOn(n, v)) return num(0);
    switch (n.k) {
      case 'num':
      case 'const':
        return num(0);
      case 'var':
        return num(n.name === v ? 1 : 0);
      case 'neg':
        return { k: 'neg', a: d(n.a) };
      case 'bin': {
        const { a, b } = n;
        switch (n.op) {
          case '+': return B('+', d(a), d(b));
          case '-': return B('-', d(a), d(b));
          case '*': return B('+', B('*', d(a), b), B('*', a, d(b)));
          case '/': return B('/', B('-', B('*', d(a), b), B('*', a, d(b))), B('^', b, num(2)));
          case '^': return dPow(a, b);
        }
        break;
      }
      case 'call': {
        const u = n.args[0], du = d(u);
        switch (n.fn) {
          case 'sin': return B('*', C('cos', u), du);
          case 'cos': return { k: 'neg', a: B('*', C('sin', u), du) };
          case 'tan': return B('/', du, B('^', C('cos', u), num(2)));
          case 'asin': return B('/', du, C('sqrt', B('-', num(1), B('^', u, num(2)))));
          case 'acos': return { k: 'neg', a: B('/', du, C('sqrt', B('-', num(1), B('^', u, num(2))))) };
          case 'atan': return B('/', du, B('+', num(1), B('^', u, num(2))));
          case 'sinh': return B('*', C('cosh', u), du);
          case 'cosh': return B('*', C('sinh', u), du);
          case 'tanh': return B('/', du, B('^', C('cosh', u), num(2)));
          case 'exp': return B('*', C('exp', u), du);
          case 'ln': return B('/', du, u);
          case 'log': return B('/', du, B('*', u, num(Math.LN10)));
          case 'sqrt': return B('/', du, B('*', num(2), C('sqrt', u)));
          case 'cbrt': return B('/', du, B('*', num(3), B('^', C('cbrt', u), num(2))));
          case 'abs':
            piecewise = true;
            return B('*', C('sign', u), du);
          case 'sign':
          case 'floor':
            piecewise = true;
            return num(0);
          case 'min':
          case 'max': {
            // min = (u + w − |u − w|)/2, max = (u + w + |u − w|)/2
            piecewise = true;
            const w = n.args[1], dw = d(w);
            const s = B('*', C('sign', B('-', u, w)), B('-', du, dw));
            return B('/', n.fn === 'max' ? B('+', B('+', du, dw), s) : B('-', B('+', du, dw), s), num(2));
          }
          case 'pow':
            return dPow(u, n.args[1]);
        }
      }
    }
    throw new Error('unreachable');
  }
  function dPow(a: Node, b: Node): Node {
    if (!dependsOn(b, v)) {
      // b·a^(b−1)·a′, with b − 1 written as a fraction when b is one, so a^(1/3)′ keeps the odd-root rule
      // b·a^(b−1)·a′; when b is an exact fraction, b − 1 stays exact (simplify folds it, or keeps it as written)
      return B('*', B('*', b, B('^', a, B('-', b, num(1)))), d(a));
    }
    if (a.k === 'const' && a.name === 'e') return B('*', B('^', a, b), d(b)); // (e^u)′ = e^u·u′
    if (!dependsOn(a, v)) return B('*', B('*', B('^', a, b), C('ln', a)), d(b)); // a^b·ln a·b′
    return B('*', B('^', a, b), B('+', B('*', d(b), C('ln', a)), B('/', B('*', b, d(a)), a)));
  }
  const ast = simplify(d(n));
  return { ast, piecewise };
}

// ---------------------------------------------------------------- exact values and smoothness at a point

/**
 * The exact value of a formula at exact rational inputs, where rational arithmetic gives it: numbers (a literal is the
 * decimal it was typed as: 0.1 is 1/10), + − × ÷, integer powers (and a power whose base is 0 or 1), abs, sign, floor,
 * min, max, sqrt of a perfect square, and the functions at the arguments where they are rational (sin, tan, asin, atan,
 * sinh, tanh, cbrt at 0 are 0; cos, cosh, exp at 0 are 1; ln and log at 1 are 0). Null where the value is not rational
 * by these rules (π, e, cos 1, √2, 2^(1/2)) or not defined (÷ 0). A page writes “=” only for such a value — a double
 * cannot tell: cos(10⁻¹⁰) is 1 in floating point (E25, Codex design review 2026-10-08). Literals are read from the text
 * they were typed as (`src`); a value `simplify` folded is exact only when it is an integer, so use it on a formula as
 * parsed (a folded cos(10⁻¹⁰) is the integer 1). For a derivative's exact value use `exactDerivative`.
 */
/** A number node's exact value: the literal as typed (0.1 is 1/10, 0.99999999999999999 is not 1), an integer made by
 *  the parser or `simplify` as it is; any other folded value is not exact (null). */
function literalQ(m: { v: number; src?: string }): Q | null {
  if (m.src !== undefined) {
    const t = m.src.match(/^(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/);
    if (t && (t[1] || t[2])) {
      const e = Number(t[3] ?? 0), frac = t[2] ?? '';
      if (Math.abs(e) > 400 || (t[1] + frac).length > 400) return null;
      const n = BigInt((t[1] || '0') + frac), k = e - frac.length;
      return k >= 0 ? Q.of(n * 10n ** BigInt(k)) : Q.of(n, 10n ** BigInt(-k));
    }
    return null;
  }
  return Number.isSafeInteger(m.v) ? Q.of(m.v) : null;
}

export function exactValue(n: Node, env: Record<string, Q>): Q | null {
  // past this many bits in a numerator or denominator the value is not one a page would print exactly: give up (nested
  // powers such as (x^999)^999 would otherwise build numbers of millions of digits; Codex review 2026-10-08)
  const MAX_BITS = 4096;
  const bits = (q: Q) => (q.n < 0n ? -q.n : q.n).toString(2).length + q.d.toString(2).length;
  try { return evalExact(n, env, MAX_BITS, bits); } catch { return null; }
}
function evalExact(n: Node, env: Record<string, Q>, MAX_BITS: number, bits: (q: Q) => number): Q | null {
  const ZERO = Q.ZERO, ONE = Q.ONE;
  const ok = (q: Q | null) => (q && bits(q) <= MAX_BITS ? q : null);
  const power = (a: Q, b: Q): Q | null => {
    if (b.isInteger()) {
      const k = Number(b.n);
      if (a.isZero() && k < 0) return null;
      if (a.isZero() || a.eq(ONE)) return k === 0 ? ONE : a;
      return Math.abs(k) * bits(a) > MAX_BITS ? null : ok(a.pow(k));
    }
    if (a.eq(ONE)) return ONE;
    if (a.isZero() && b.sign() > 0) return ZERO;
    return null;
  };
  const ev = (m: Node): Q | null => {
    switch (m.k) {
      case 'num': return literalQ(m);
      case 'var': return env[m.name] ?? null;
      case 'const': return null;
      case 'neg': { const a = ev(m.a); return a && a.neg(); }
      case 'bin': {
        const a = ev(m.a);
        if (m.op === '^') {
          const b = ev(m.b);
          return a && b ? power(a, b) : null;
        }
        const b = ev(m.b);
        if (!a || !b) return null;
        if (m.op === '+') return ok(a.add(b));
        if (m.op === '-') return ok(a.sub(b));
        if (m.op === '*') return ok(a.mul(b));
        return b.isZero() ? null : ok(a.div(b));
      }
      case 'call': {
        const args = m.args.map(ev);
        if (args.some((x) => !x)) return null;
        const u = args[0]!;
        switch (m.fn) {
          case 'abs': return u.abs();
          case 'sign': return Q.of(u.sign());
          case 'floor': return Q.of(u.floor());
          case 'min': return Q.min(u, args[1]!);
          case 'max': return Q.max(u, args[1]!);
          case 'sqrt': return u.sign() < 0 ? null : u.sqrtExact();
          case 'pow': return power(u, args[1]!);
          case 'sin': case 'tan': case 'asin': case 'atan': case 'sinh': case 'tanh': case 'cbrt': return u.isZero() ? ZERO : null;
          case 'cos': case 'cosh': case 'exp': return u.isZero() ? ONE : null;
          case 'ln': case 'log': return u.eq(ONE) ? ZERO : null;
          default: return null;
        }
      }
    }
  };
  return ev(n);
}

/**
 * A formula's value and its derivative in `v` at exact rational inputs, by forward-mode differentiation over the
 * rationals on the formula as parsed (no floating-point folding): every step exact or the whole is null — the same
 * rules as `exactValue` for the values (sin, tan, asin, atan, sinh, tanh, exp, cos, cosh at 0; ln at 1; sqrt of a
 * nonzero perfect square; integer powers), and null at a kink (abs or sign at 0, floor at an integer, min or max at a
 * tie), where the derivative of a constant π or e would be needed, or where the derivative is not rational (log's
 * 1/ln 10). E25's divergence is “=” only through this (Codex review 2026-10-08).
 */
export function exactDerivative(n: Node, v: string, env: Record<string, Q>): { value: Q; d: Q } | null {
  const MAX_BITS = 4096;
  const bits = (q: Q) => (q.n < 0n ? -q.n : q.n).toString(2).length + q.d.toString(2).length;
  const ZERO = Q.ZERO, ONE = Q.ONE;
  type Dual = { f: Q; d: Q } | null;
  const ok = (x: Dual): Dual => (x && bits(x.f) <= MAX_BITS && bits(x.d) <= MAX_BITS ? x : null);
  const pow = (a: NonNullable<Dual>, b: NonNullable<Dual>): Dual => {
    if (!b.d.isZero()) return a.f.eq(ONE) && a.d.isZero() ? { f: ONE, d: ZERO } : null; // 1^u, and nothing else with u in v
    if (b.f.isInteger()) {
      const k = Number(b.f.n);
      if (k === 0) return { f: ONE, d: ZERO };
      if (a.f.isZero()) return k < 0 ? null : k === 1 ? a : { f: ZERO, d: ZERO };
      if (Math.abs(k) * bits(a.f) > MAX_BITS) return null;
      const fk1 = a.f.pow(k - 1);
      return ok({ f: fk1.mul(a.f), d: Q.of(k).mul(fk1).mul(a.d) });
    }
    return a.f.eq(ONE) ? { f: ONE, d: b.f.mul(a.d) } : null;
  };
  const ev = (m: Node): Dual => {
    switch (m.k) {
      case 'num': { const q = literalQ(m); return q && { f: q, d: ZERO }; }
      case 'var': { const x = env[m.name]; return x ? { f: x, d: m.name === v ? ONE : ZERO } : null; }
      case 'const': return null;
      case 'neg': { const a = ev(m.a); return a && { f: a.f.neg(), d: a.d.neg() }; }
      case 'bin': {
        const a = ev(m.a), b = ev(m.b);
        if (!a || !b) return null;
        switch (m.op) {
          case '+': return ok({ f: a.f.add(b.f), d: a.d.add(b.d) });
          case '-': return ok({ f: a.f.sub(b.f), d: a.d.sub(b.d) });
          case '*': return ok({ f: a.f.mul(b.f), d: a.d.mul(b.f).add(a.f.mul(b.d)) });
          case '/': return b.f.isZero() ? null : ok({ f: a.f.div(b.f), d: a.d.mul(b.f).sub(a.f.mul(b.d)).div(b.f.mul(b.f)) });
          case '^': return pow(a, b);
        }
        return null;
      }
      case 'call': {
        const args = m.args.map(ev);
        if (args.some((x) => !x)) return null;
        const u = args[0]!, z = u.f.isZero();
        switch (m.fn) {
          case 'sin': case 'tan': case 'asin': case 'atan': case 'sinh': case 'tanh': return z ? { f: ZERO, d: u.d } : null;
          case 'cos': case 'cosh': return z ? { f: ONE, d: ZERO } : null;
          case 'exp': return z ? { f: ONE, d: u.d } : null;
          case 'ln': return u.f.eq(ONE) ? { f: ZERO, d: u.d } : null;
          case 'log': return u.f.eq(ONE) && u.d.isZero() ? { f: ZERO, d: ZERO } : null;
          case 'sqrt': { if (u.f.sign() <= 0) return null; const r = u.f.sqrtExact(); return r ? ok({ f: r, d: u.d.div(r.mul(Q.of(2))) }) : null; }
          case 'cbrt': return u.f.eq(ONE) ? { f: ONE, d: u.d.div(Q.of(3)) } : u.f.eq(ONE.neg()) ? { f: ONE.neg(), d: u.d.div(Q.of(3)) } : null;
          case 'abs': return z ? null : { f: u.f.abs(), d: u.f.sign() < 0 ? u.d.neg() : u.d };
          case 'sign': return z ? null : { f: Q.of(u.f.sign()), d: ZERO };
          case 'floor': return u.f.isInteger() ? null : { f: Q.of(u.f.floor()), d: ZERO };
          case 'min': case 'max': {
            const w = args[1]!, c = u.f.cmp(w.f);
            if (c === 0) return null;
            return (m.fn === 'min') === (c < 0) ? u : w;
          }
          case 'pow': return pow(u, args[1]!);
        }
        return null;
      }
    }
  };
  try { const r = ev(n); return r ? { value: r.f, d: r.d } : null; } catch { return null; }
}

/**
 * False when the point sits exactly where a piecewise function in the formula bends or jumps — abs or sign at a zero
 * argument, floor at an integer, min or max at a tie — so a derivative from `differentiate` (which takes abs′ = sign,
 * sign′ = floor′ = 0, a tie's average slope) does not hold there. Arguments are compared exactly when `exactValue`
 * can, else as doubles. An optional exact environment supplies the original rational inputs instead of inferring
 * them from printed doubles; other callers keep the existing decimal interpretation. Elsewhere true: whether the
 * formula is defined at all is the caller's to check (diagnose accepts the same exact environment).
 */
export function smoothAt(n: Node, env: Record<string, number>, exactEnv?: Record<string, Q>): boolean {
  const qenv: Record<string, Q> = { ...exactEnv };
  for (const [k, v] of Object.entries(env)) { const q = qenv[k] ?? (Number.isFinite(v) ? parseExact(String(v), 17) ?? Q.fromDouble(v) : null); if (q) qenv[k] = q; }
  const val = (m: Node): { q: Q | null; v: number } => ({ q: exactValue(m, qenv), v: compile(m)(env) });
  const walk = (m: Node): boolean => {
    switch (m.k) {
      case 'num': case 'var': case 'const': return true;
      case 'neg': return walk(m.a);
      case 'bin': return walk(m.a) && walk(m.b);
      case 'call': {
        if (!m.args.every(walk)) return false;
        if (m.fn === 'abs' || m.fn === 'sign') { const a = val(m.args[0]); return a.q ? !a.q.isZero() : a.v !== 0; }
        if (m.fn === 'floor') { const a = val(m.args[0]); return a.q ? !a.q.isInteger() : !Number.isInteger(a.v); }
        if (m.fn === 'min' || m.fn === 'max') { const a = val(m.args[0]), b = val(m.args[1]); return a.q && b.q ? !a.q.eq(b.q) : a.v !== b.v; }
        return true;
      }
    }
  };
  return walk(n);
}

// ---------------------------------------------------------------- printing

const PREC: Record<string, number> = { '+': 1, '-': 1, '*': 2, '/': 2, neg: 3, '^': 4 };
const precOf = (n: Node) => (n.k === 'bin' ? PREC[n.op] : n.k === 'neg' || (n.k === 'num' && n.v < 0) ? PREC.neg : 9);
/** For display (TeX): 12 significant digits. */
const fmtNum = (v: number) => (Number.isInteger(v) && Math.abs(v) < 1e21 ? String(v) : String(+v.toPrecision(12)));

/**
 * Plain text that `parse` reads back to the same tree, up to the sign of numeric literals: numbers in their shortest
 * round-trip form, explicit *, and parentheses wherever the grouping differs from the parser's default, so evaluation
 * order (and with it every rounding) is unchanged. Long derivatives can exceed the parser's default length limit:
 * read them back with `parse(text, { maxLength: Infinity, maxDepth: Infinity })`.
 */
export function toText(n: Node): string {
  switch (n.k) {
    case 'num': return String(n.v); // shortest round-trip form; a negative number is bracketed by its parent
    case 'var': return n.name;
    case 'const': return n.name === 'pi' ? 'π' : 'e';
    case 'neg': {
      const a = toText(n.a);
      return `-${precOf(n.a) <= PREC.neg ? `(${a})` : a}`;
    }
    case 'bin': {
      const p = PREC[n.op];
      const left = n.op === '^' ? precOf(n.a) <= p : precOf(n.a) < p;
      const right = n.op === '^' ? precOf(n.b) < p : precOf(n.b) <= p; // a + (b + c) is not (a + b) + c in floating point
      const sa = toText(n.a), sb = toText(n.b);
      const l = left ? `(${sa})` : sa, r = right ? `(${sb})` : sb;
      return n.op === '^' ? `${l}^${r}` : `${l} ${n.op} ${r}`;
    }
    case 'call': return `${n.fn}(${n.args.map(toText).join(', ')})`;
  }
}

const TEX_FN: Partial<Record<Fn, string>> = {
  sin: '\\sin', cos: '\\cos', tan: '\\tan', asin: '\\arcsin', acos: '\\arccos', atan: '\\arctan', sinh: '\\sinh', cosh: '\\cosh',
  tanh: '\\tanh', exp: '\\exp', ln: '\\ln', log: '\\log_{10}', min: '\\min', max: '\\max', sign: '\\operatorname{sgn}', floor: '\\operatorname{floor}',
};

/** Does the printed form begin with a number (so a product needs a visible dot before it)? */
const leadsWithNumber = (n: Node): boolean => n.k === 'num' || (n.k === 'bin' && n.op !== '/' && leadsWithNumber(n.a));

/** KaTeX for the formula (fractions for division, raised exponents, √, |…|). */
export function toTex(n: Node): string {
  const paren = (s: string) => `\\left(${s}\\right)`;
  switch (n.k) {
    case 'num': return fmtNum(n.v);
    case 'var': return n.name;
    case 'const': return n.name === 'pi' ? '\\pi' : 'e';
    case 'neg': {
      const a = toTex(n.a);
      return `-${precOf(n.a) <= PREC.neg ? paren(a) : a}`;
    }
    case 'bin': {
      const sa = toTex(n.a), sb = toTex(n.b);
      switch (n.op) {
        case '/': return `\\frac{${sa}}{${sb}}`;
        case '^': return `${precOf(n.a) <= PREC['^'] || n.a.k === 'call' ? paren(sa) : sa}^{${sb}}`;
        case '*': {
          const l = precOf(n.a) < PREC['*'] ? paren(sa) : sa;
          const r = precOf(n.b) < PREC['*'] || precOf(n.b) === PREC.neg ? paren(sb) : sb;
          return `${l}${leadsWithNumber(n.b) && r === sb ? ' \\cdot ' : ' '}${r}`;
        }
        default: {
          const r = precOf(n.b) === PREC.neg || (n.op === '-' && precOf(n.b) <= PREC['-']) ? paren(sb) : sb;
          return `${sa} ${n.op} ${r}`;
        }
      }
    }
    case 'call': {
      const args = n.args.map(toTex);
      if (n.fn === 'sqrt') return `\\sqrt{${args[0]}}`;
      if (n.fn === 'cbrt') return `\\sqrt[3]{${args[0]}}`;
      if (n.fn === 'abs') return `\\left|${args[0]}\\right|`;
      if (n.fn === 'pow') return `${paren(args[0])}^{${args[1]}}`;
      return `${TEX_FN[n.fn]}${paren(args.join(', '))}`;
    }
  }
}
