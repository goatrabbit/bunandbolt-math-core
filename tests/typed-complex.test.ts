// typedComplex (src/lib/ui/typed.ts): a typed complex number in rectangular, polar or exponential form, every part read
// as the site reads numbers, exact where the text is (2026-10-07, A15 complex plane).
import { describe, expect, it } from 'vitest';
import { typedComplex, type TypedComplex } from '../src/lib/ui/typed';

type Ok = Extract<TypedComplex, { ok: true }>;
const ok = (t: string) => {
  const r = typedComplex(t);
  if (!r.ok) throw new Error(`refused ${t}: ${r.reason}`);
  return r as Ok;
};
const rect = (t: string) => {
  const r = ok(t);
  if (r.form !== 'rect') throw new Error(`${t} read as ${r.form}`);
  return [r.re, r.im] as const;
};
const polar = (t: string) => {
  const r = ok(t);
  if (r.form !== 'polar') throw new Error(`${t} read as ${r.form}`);
  return r;
};

describe('rectangular', () => {
  it('reads a ± bi, exactly for decimals and fractions', () => {
    const [a, b] = rect('2+i');
    expect([a.q!.toString(), b.q!.toString()]).toEqual(['2', '1']);
    expect(rect('2 - 0.5i').map((x) => x.q!.toString())).toEqual(['2', '-1/2']);
    expect(rect('-i').map((x) => x.q!.toString())).toEqual(['0', '-1']);
    expect(rect('3i').map((x) => x.q!.toString())).toEqual(['0', '3']);
    expect(rect('-2.5').map((x) => x.q!.toString())).toEqual(['-5/2', '0']);
    expect(rect('i/2').map((x) => x.q!.toString())).toEqual(['0', '1/2']);
    expect(rect('2·i').map((x) => x.q!.toString())).toEqual(['0', '2']);
    expect(rect('１＋２ｉ').map((x) => x.q!.toString())).toEqual(['1', '2']);
    expect(rect('1e-3+2e-3i').map((x) => x.q!.toString())).toEqual(['1/1000', '1/500']);
  });
  it('a root or π part is numeric (its value, no rational)', () => {
    const [a, b] = rect('1/2+√3/2i');
    expect(a.q!.toString()).toBe('1/2');
    expect(b.q).toBeNull();
    expect(b.value).toBeCloseTo(Math.sqrt(3) / 2, 15);
    expect(rect('1/2 + i√3/2')[1].value).toBeCloseTo(Math.sqrt(3) / 2, 15);
    const [e, p] = rect('e+πi');
    expect(e.value).toBe(Math.E);
    expect(p.value).toBe(Math.PI);
  });
  it('refuses what is not one number', () => {
    for (const t of ['2+3', 'i+i', '2+i+1', 'ii', '2i3', 'x+i', '', '1,2,3+i', '2++i', '1e-999+i']) expect(typedComplex(t).ok).toBe(false);
    expect(typedComplex('').ok === false && (typedComplex('') as { reason: string }).reason).toBe('empty');
  });
});

describe('polar', () => {
  it('r∠θ with θ in degrees or with π: an exact fraction of a turn', () => {
    const w = polar('1.5∠40°');
    expect(w.r.q!.toString()).toBe('3/2');
    expect(w.r.sq!.toString()).toBe('9/4');
    expect(w.angle.turn!.toString()).toBe('1/9');
    expect(polar('2∠π/3').angle.turn!.toString()).toBe('1/6');
    expect(polar('2<120deg').angle.turn!.toString()).toBe('1/3');
    expect(polar('1∠-90°').angle.turn!.toString()).toBe('-1/4');
    expect(polar('1∠400°').angle.turn!.toString()).toBe('10/9'); // not reduced: the page checks its range
  });
  it('a bare angle is radians, numeric', () => {
    const w = polar('1∠-1.2');
    expect(w.angle.turn).toBeNull();
    expect(w.angle.value).toBe(-1.2);
  });
  it('a root modulus has an exact square', () => {
    const w = polar('√2∠45°');
    expect(w.r.q).toBeNull();
    expect(w.r.sq!.toString()).toBe('2');
  });
  it('refuses a negative modulus and a broken angle', () => {
    for (const t of ['-2∠30°', '2∠', '∠30°', '2∠30°∠1', '2∠x']) expect(typedComplex(t).ok).toBe(false);
  });
});

describe('exponential', () => {
  it('r·e^(iθ) in the usual spellings', () => {
    expect(polar('e^(iπ/3)').angle.turn!.toString()).toBe('1/6');
    expect(polar('e^(iπ/3)').r.q!.toString()).toBe('1');
    const w = polar('2e^(iπ/6)');
    expect(w.r.q!.toString()).toBe('2');
    expect(w.angle.turn!.toString()).toBe('1/12');
    expect(polar('e^{2πi/5}').angle.turn!.toString()).toBe('1/5');
    expect(polar('3·e^(i90°)').angle.turn!.toString()).toBe('1/4');
    expect(polar('e^(i)').angle.value).toBe(1); // e^{i·1}: radians, numeric
    expect(polar('e^iπ').angle.turn!.toString()).toBe('1/2');
    expect(polar('e^(-iπ/2)').angle.turn!.toString()).toBe('-1/4');
  });
  it('refuses an exponent without i', () => {
    for (const t of ['e^(π/3)', 'e^2', '-e^(iπ)']) expect(typedComplex(t).ok).toBe(false);
  });
  it('e alone is Euler’s number, a real', () => {
    expect(rect('e')[0].value).toBe(Math.E);
  });
});

describe('exponents that are not one angle (Codex implementation review 1)', () => {
  it('refuses a real term beside iθ', () => {
    for (const t of ['e^(i+π/3)', 'e^(i-1)', 'e^(π/3i)', 'e^(i+1)']) expect(typedComplex(t).ok, t).toBe(false);
  });
  it('reads i/2 as half a radian and a leading +', () => {
    expect(polar('e^(i/2)').angle.value).toBe(0.5);
    expect(polar('+e^(iπ/3)').angle.turn!.toString()).toBe('1/6');
  });
});
