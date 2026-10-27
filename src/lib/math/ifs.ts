// Iterated function systems in the plane (first used by WB08, the chaos game): finitely many contracting affine maps,
// the one non-empty compact set they fix (Hutchinson 1981: K = f₁(K) ∪ … ∪ f_N(K)), its copies f_w(K) named by words,
// and the facts stated about regular polygons. Pure functions in double precision.
//
// Conventions
// - An affine map is [a, b, c, d, e, f]: x′ = a·x + b·y + e, y′ = c·x + d·y + f.
// - Rolls i₁, i₂, … move a point to xₙ = f_{iₙ}(xₙ₋₁). A word w = w₁w₂…w_k names the copy f_{w₁} ∘ f_{w₂} ∘ … ∘ f_{w_k}:
//   w₁ is the last roll, w_k the earliest of the k ("colour by the j-th last roll" colours a copy by w_j).
// - A restriction says which roll may follow which: allowed[prev][next]. 'norep': not the same map twice in a row;
//   'notnext': not a neighbour (cyclically ±1) of the previous one — meant for five or more points.

export type Affine = [number, number, number, number, number, number];
export type V2 = [number, number];
export type Restriction = 'any' | 'norep' | 'notnext';
export interface Box { x0: number; y0: number; x1: number; y1: number }

export const IDENTITY: Affine = [1, 0, 0, 1, 0, 0];
export const apply = (m: Affine, x: number, y: number): V2 => [m[0] * x + m[1] * y + m[4], m[2] * x + m[3] * y + m[5]];
/** f ∘ g (g first). */
export function compose(f: Affine, g: Affine): Affine {
  return [
    f[0] * g[0] + f[1] * g[2], f[0] * g[1] + f[1] * g[3],
    f[2] * g[0] + f[3] * g[2], f[2] * g[1] + f[3] * g[3],
    f[0] * g[4] + f[1] * g[5] + f[4], f[2] * g[4] + f[3] * g[5] + f[5],
  ];
}
export const det = (m: Affine) => m[0] * m[3] - m[1] * m[2];
/** The largest singular value of the linear part: the most the map stretches a distance (a contraction: < 1). */
export function sigmaMax(m: Affine): number {
  const s = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3], d = det(m);
  return Math.sqrt((s + Math.sqrt(Math.max(0, s * s - 4 * d * d))) / 2);
}
/** The jump a fraction r = 1 − s of the way towards v: x ↦ v + s(x − v). */
export const toward = (v: V2, s: number): Affine => [s, 0, 0, s, (1 - s) * v[0], (1 - s) * v[1]];
/** The affine map sending P₀, P₁, P₂ to Q₀, Q₁, Q₂ (null if the P are collinear). */
export function through(P: readonly V2[], Q: readonly V2[]): Affine | null {
  const ux = P[1][0] - P[0][0], uy = P[1][1] - P[0][1], vx = P[2][0] - P[0][0], vy = P[2][1] - P[0][1];
  const D = ux * vy - vx * uy;
  if (Math.abs(D) < 1e-300) return null;
  const px = Q[1][0] - Q[0][0], py = Q[1][1] - Q[0][1], qx = Q[2][0] - Q[0][0], qy = Q[2][1] - Q[0][1];
  // A·[u v] = [p q]  ⇒  A = [p q]·[u v]⁻¹
  const a = (px * vy - qx * uy) / D, b = (qx * ux - px * vx) / D, c = (py * vy - qy * uy) / D, d = (qy * ux - py * vx) / D;
  return [a, b, c, d, Q[0][0] - a * P[0][0] - b * P[0][1], Q[0][1] - c * P[0][0] - d * P[0][1]];
}

// ------------------------------------------------------------------------------------------------ regular polygons

/** The regular n-gon with circumradius 1 centred at 0, counter-clockwise: an odd n has a vertex at the top, an even n
 *  a flat bottom (and top). */
export function regularPolygon(n: number): V2[] {
  const a0 = Math.PI / 2 + (n % 2 ? 0 : Math.PI / n);
  return Array.from({ length: n }, (_, k) => [Math.cos(a0 + (2 * Math.PI * k) / n), Math.sin(a0 + (2 * Math.PI * k) / n)] as V2);
}
/** The carpet's eight points: the corners and edge midpoints of [−1, 1]², counter-clockwise from the top left (each
 *  point's neighbours are the ones before and after it along the boundary). */
export const CARPET: V2[] = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]];

/** The scale s at which the n copies s·P + (1 − s)·vᵢ of a regular n-gon just touch (the n-flake ratio):
 *  s = 1 / (2(1 + Σ_{k=1}^{⌊n/4⌋} cos(2πk/n))). n = 3, 4: 1/2; 5: (3 − √5)/2; 6: 1/3; 8: 1 − √2/2. */
export function touchingScale(n: number): number {
  let sum = 0;
  for (let k = 1; k <= Math.floor(n / 4); k++) sum += Math.cos((2 * Math.PI * k) / n);
  return 1 / (2 * (1 + sum));
}
/** The least scale at which the n copies cover the whole regular n-gon (with any roll allowed the picture is then the
 *  solid polygon). The last point covered is the centre: an odd n needs (1 − s)/s ≤ cos(π/n) (the centre seen from a
 *  vertex reaches the opposite edge), an even n needs (1 − s)/s ≤ 1 (the opposite vertex). */
export const fillScale = (n: number) => (n % 2 ? 1 / (1 + Math.cos(Math.PI / n)) : 0.5);
/** The carpet's eight copies touch at s = 1/3 (the Sierpiński carpet) and cover the square from s = 1/2. */
export const CARPET_TOUCH = 1 / 3;
export const CARPET_FILL = 1 / 2;

// ------------------------------------------------------------------------------------------------ restrictions

/** allowed[prev][next] for n points. */
export function allowedTable(n: number, rule: Restriction): boolean[][] {
  return Array.from({ length: n }, (_, p) => Array.from({ length: n }, (_, q) => {
    if (rule === 'norep') return q !== p;
    if (rule === 'notnext') return q !== (p + 1) % n && q !== (p + n - 1) % n;
    return true;
  }));
}
/** How many rolls may follow any roll (the same for every point in these rules, so it is the transition matrix's
 *  spectral radius). */
export const branching = (n: number, rule: Restriction) => (rule === 'any' ? n : rule === 'norep' ? n - 1 : n - 2);
/** Similarity dimension log m / log(1/s): the Hausdorff dimension when the copies do not overlap (open set condition;
 *  for a rule with memory, Mauldin–Williams with every state having m successors). Always an upper bound. */
export const similarityDimension = (m: number, s: number) => Math.log(m) / Math.log(1 / s);

// ------------------------------------------------------------------------------------------------ convex geometry

/** Convex hull, counter-clockwise, collinear points dropped (Andrew's monotone chain). Two points: a segment; one: a point. */
export function convexHull(pts: readonly V2[]): V2[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const uniq = p.filter((q, i) => i === 0 || q[0] !== p[i - 1][0] || q[1] !== p[i - 1][1]);
  if (uniq.length < 3) return uniq;
  const cross = (o: V2, a: V2, b: V2) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: V2[] = [], upper: V2[] = [];
  for (const q of uniq) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = uniq.length - 1; i >= 0; i--) { const q = uniq[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
export function boxOf(pts: readonly V2[]): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return { x0, y0, x1, y1 };
}
export const mapPoly = (m: Affine, poly: readonly V2[]): V2[] => poly.map(([x, y]) => apply(m, x, y));

/** How far apart two convex polygons (or collinear segments) are: the largest gap between their shadows on a separating
 *  axis (edge normals; for a segment also its direction). Positive: apart by that much; zero: touching; negative: the
 *  interiors overlap (the least penetration). */
export function gap(A: readonly V2[], B: readonly V2[]): number {
  const axes: V2[] = [];
  for (const P of [A, B]) {
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length], dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
      if (L > 0) { axes.push([-dy / L, dx / L]); if (P.length === 2) axes.push([dx / L, dy / L]); }
    }
  }
  let best = -Infinity;
  for (const [ux, uy] of axes) {
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const [x, y] of A) { const t = x * ux + y * uy; if (t < a0) a0 = t; if (t > a1) a1 = t; }
    for (const [x, y] of B) { const t = x * ux + y * uy; if (t < b0) b0 = t; if (t > b1) b1 = t; }
    best = Math.max(best, b0 - a1, a0 - b1);
  }
  return best;
}
/** The open set condition through the convex hull H of the points: the least gap between two copies fᵢ(H), in units
 *  of H's size (numerical). Positive or zero: the copies' interiors are pairwise disjoint, which for maps
 *  x ↦ vᵢ + s(x − vᵢ) with H the hull of the vᵢ is sufficient (fᵢ(int H) ⊂ int H; a rule with memory keeps the same
 *  open sets). A value within rounding of zero cannot tell touching from overlapping. */
export function copiesGap(maps: readonly Affine[], hull: readonly V2[]): number {
  const b = boxOf(hull), size = Math.max(b.x1 - b.x0, b.y1 - b.y0, 1e-300);
  const imgs = maps.map((m) => mapPoly(m, hull));
  let least = Infinity;
  for (let i = 0; i < imgs.length; i++) for (let j = i + 1; j < imgs.length; j++) least = Math.min(least, gap(imgs[i], imgs[j]));
  return least / size;
}

/** A convex polygon that certainly contains the attractor, near its convex hull. Start from a disc B(c, R) mapped
 *  into itself (R ≥ |fᵢ(c) − c| / (1 − σᵢ) for every i), then apply C ↦ hull(∪ fᵢ(C)) `iters` times: each step keeps
 *  K inside (K = ∪ fᵢ(K)) and is a contraction towards hull(K) by max σᵢ. */
export function enclosingHull(maps: readonly Affine[], iters = 48): V2[] {
  const fixed = maps.map(fixedPoint).filter((p): p is V2 => !!p);
  const c: V2 = fixed.length ? [fixed.reduce((s, p) => s + p[0], 0) / fixed.length, fixed.reduce((s, p) => s + p[1], 0) / fixed.length] : [0, 0];
  let R = 0;
  for (const m of maps) { const [x, y] = apply(m, c[0], c[1]); R = Math.max(R, Math.hypot(x - c[0], y - c[1]) / (1 - sigmaMax(m))); }
  R = Math.max(R, 1e-12);
  const sides = 24, Rc = R / Math.cos(Math.PI / sides);
  let poly: V2[] = Array.from({ length: sides }, (_, k) => [c[0] + Rc * Math.cos((2 * Math.PI * k) / sides), c[1] + Rc * Math.sin((2 * Math.PI * k) / sides)] as V2);
  for (let t = 0; t < iters; t++) poly = convexHull(maps.flatMap((m) => mapPoly(m, poly)));
  return poly;
}
/** The fixed point of a contraction (null if I − A is singular, which a contraction never is). */
export function fixedPoint(m: Affine): V2 | null {
  const a = 1 - m[0], b = -m[1], c = -m[2], d = 1 - m[3], D = a * d - b * c;
  if (Math.abs(D) < 1e-300) return null;
  return [(d * m[4] - b * m[5]) / D, (a * m[5] - c * m[4]) / D];
}

// ------------------------------------------------------------------------------------------------ odds

/** Odds weighted by area: pᵢ = max(floor, c·|det Aᵢ|) with c chosen so that they sum to 1 (a degenerate map, like
 *  the fern's stem, gets exactly the floor). All areas zero, or a floor too large to give: equal odds. */
export function oddsByArea(maps: readonly Affine[], floor = 0.01): number[] {
  const n = maps.length, a = maps.map((m) => Math.abs(det(m)));
  if (!(a.some((v) => v > 0)) || floor * n >= 1) return maps.map(() => 1 / n);
  const low = new Set<number>();
  for (;;) {
    const rest = a.reduce((s, v, i) => (low.has(i) ? s : s + v), 0);
    const c = (1 - floor * low.size) / rest;
    const add = a.map((v, i) => i).filter((i) => !low.has(i) && c * a[i] < floor);
    if (!add.length) return a.map((v, i) => (low.has(i) ? floor : c * v));
    for (const i of add) low.add(i);
  }
}
/** Rolls not drawn before the picture is trusted: the start's distance to K, shrunk by σ per roll, falls below `eps`
 *  of K's size (the start is within K's size of K). Capped. */
export const warmup = (sMax: number, eps: number, cap = 400) => Math.min(cap, Math.max(1, Math.ceil(Math.log(eps) / Math.log(sMax))));

// ------------------------------------------------------------------------------------------------ copies (words)

/** All allowed words of length k as composed maps: copies.maps holds 6 numbers per copy; first[3i + j] is w_{j+1}
 *  (255 when the word is shorter); parent[i] is the copy of length k − 1 it was made from (f_{w₁} ∘ parent), so the
 *  copy machine can fly the previous picture's copies into place. Words of length k + 1 come from words of length k by
 *  a new last roll w₁ allowed after the old w₁. */
export interface Copies { k: number; count: number; maps: Float64Array; first: Uint8Array; parent: Uint32Array }
export function copiesOf(maps: readonly Affine[], allowed: readonly (readonly boolean[])[], k: number): Copies {
  let cur: Copies = { k: 0, count: 1, maps: new Float64Array(IDENTITY), first: new Uint8Array([255, 255, 255]), parent: new Uint32Array(1) };
  for (let level = 1; level <= k; level++) cur = nextCopies(maps, allowed, cur);
  return cur;
}
export function nextCopies(maps: readonly Affine[], allowed: readonly (readonly boolean[])[], prev: Copies): Copies {
  const n = maps.length;
  const idx: number[] = [], outer: number[] = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < prev.count; j++) {
    const w1 = prev.first[3 * j];
    if (w1 === 255 || allowed[w1][i]) { idx.push(j); outer.push(i); }
  }
  const count = idx.length, out = new Float64Array(6 * count), first = new Uint8Array(3 * count), parent = new Uint32Array(count);
  const g: Affine = [0, 0, 0, 0, 0, 0];
  for (let c = 0; c < count; c++) {
    const j = idx[c], i = outer[c];
    for (let t = 0; t < 6; t++) g[t] = prev.maps[6 * j + t];
    out.set(compose(maps[i], g), 6 * c);
    first[3 * c] = i; first[3 * c + 1] = prev.first[3 * j]; first[3 * c + 2] = prev.first[3 * j + 1];
    parent[c] = j;
  }
  return { k: prev.k + 1, count, maps: out, first, parent };
}
/**
 * The copies refined until each is small: a word is replaced by its children while its copy is larger than `eps` (the
 * largest singular value of its linear part, so of the start shape's size). For a map that shrinks slowly (the fern's
 * body, 0.85 a round) whole rounds would need dozens of them; this keeps splitting only the large copies. The words have
 * mixed lengths (a cut), so it is not "round k". Null when there would be more than `limit`. first: as copiesOf;
 * parent: unused (0), the copies are drawn in place.
 */
export function refineCopies(maps: readonly Affine[], allowed: readonly (readonly boolean[])[], eps: number, limit: number): Copies | null {
  const out: number[] = [], first: number[] = [];
  const stack: { m: Affine; f: [number, number, number]; inner: number; depth: number }[] = [{ m: IDENTITY, f: [255, 255, 255], inner: -1, depth: 0 }];
  while (stack.length) {
    const it = stack.pop()!;
    if (sigmaMax(it.m) <= eps || it.depth >= 60) {
      out.push(...it.m);
      first.push(...it.f);
      if (out.length / 6 > limit) return null;
      continue;
    }
    for (let i = maps.length - 1; i >= 0; i--) {
      if (it.inner >= 0 && !allowed[i][it.inner]) continue;
      const f = [...it.f] as [number, number, number];
      if (it.depth < 3) f[it.depth] = i;
      stack.push({ m: compose(it.m, maps[i]), f, inner: i, depth: it.depth + 1 });
    }
  }
  const count = out.length / 6;
  return { k: -1, count, maps: Float64Array.from(out), first: Uint8Array.from(first), parent: new Uint32Array(count) };
}
/** How many words of length k: n·m^(k−1) (every point has m successors). */
export const copyCount = (n: number, m: number, k: number) => (k === 0 ? 1 : n * m ** (k - 1));
/** The largest k with at most `limit` copies (and at most `kMax`). */
export function maxCopies(n: number, m: number, limit: number, kMax = 30): number {
  let k = 0;
  while (k < kMax && copyCount(n, m, k + 1) <= limit) k++;
  return k;
}

// ------------------------------------------------------------------------------------------------ zooming by address

/** A copy seen in a window. q: its weight (the product of the odds along the word); inner: w_k (−1 for the empty word),
 *  which decides which earlier rolls it may follow under a restriction; first: w₁ w₂ w₃ (255 past the word's end). */
export interface Seen { m: Affine; q: number; depth: number; inner: number; first: [number, number, number]; size: number }
/**
 * The copies f_w(K) to draw a window from, coarse enough to be few and fine enough to fill it: starting from the empty
 * word, a word whose copy (of the enclosing hull H) misses the window is dropped, one larger than `size` is replaced by
 * its children w·i (an earlier roll i; under a restriction only those i that w_k may follow), all such words at once,
 * until none is larger or there would be more than `cap` (then those are kept as they are). The words kept form a cut, so the
 * invariant measure is Σ q_w·(f_w)_*μ over them plus copies outside the window: drawing f_w(y) for orbit points y,
 * w drawn with probability ∝ q_w among the words a point's last roll allows, gives the window's true relative density.
 * odds: per map (any rule), or null for a restriction (each of the m successors 1/m).
 */
export function seenCopies(maps: readonly Affine[], odds: readonly number[] | null, allowed: readonly (readonly boolean[])[], hull: readonly V2[], view: Box, size: number, cap = 2048, maxDepth = 60): Seen[] {
  const n = maps.length, m = odds ? n : allowed[0].filter(Boolean).length;
  const hits = (w: Affine) => {
    const b = boxOf(mapPoly(w, hull));
    return { hit: b.x1 >= view.x0 && b.x0 <= view.x1 && b.y1 >= view.y0 && b.y0 <= view.y1, size: Math.max(b.x1 - b.x0, b.y1 - b.y0) };
  };
  type Item = Seen;
  const root = hits(IDENTITY);
  if (!root.hit) return [];
  const done: Item[] = [];
  let open: Item[] = [{ m: IDENTITY, q: 1, depth: 0, inner: -1, first: [255, 255, 255], size: root.size }];
  while (open.length) {
    const big = open.filter((it) => it.size > size && it.depth < maxDepth);
    const small = open.filter((it) => !(it.size > size && it.depth < maxDepth));
    if (!big.length) { done.push(...small); break; }
    const kids: Item[] = [];
    for (const it of big) for (let i = 0; i < n; i++) {
      if (it.inner >= 0 && !allowed[i][it.inner]) continue;
      const w = compose(it.m, maps[i]), h = hits(w);
      if (!h.hit) continue;
      const first = [...it.first] as [number, number, number];
      if (it.depth < 3) first[it.depth] = i;
      kids.push({ m: w, q: it.q * (odds ? odds[i] : 1 / m), depth: it.depth + 1, inner: i, first, size: h.size });
    }
    done.push(...small);
    if (done.length + kids.length > cap) { done.push(...big); break; }
    open = kids;
  }
  return done;
}
