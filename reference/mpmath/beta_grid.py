# Reference grid for the regularized incomplete beta I_x(a, b) and 1 − I_x(a, b) (tests/fixtures/beta-mpmath.json).
# Integer a, b: exact binomial sums; a, b ≤ 1000: mpmath betainc; larger: the positive hypergeometric series at 40 digits
# on the side with z ≤ 1/2 (the other side kept only when it is ≥ 1e-25, as 1 − series). The quadrature computed first
# is overridden in every case (it is kept for comparing by hand; it is poor at endpoint singularities). Run: PYTHONPATH=<site-packages with mpmath 1.3> python3 beta_grid.py > beta-mpmath.json
from mpmath import mp, mpf, quad, log, exp, loggamma, betainc, binomial, fsum
import json
mp.dps = 40
def lnB(a, b): return loggamma(a) + loggamma(b) - loggamma(a + b)
def side(a, b, lo, hi, pts):
    L = lnB(a, b)
    f = lambda t: exp((a - 1) * log(t) + (b - 1) * log(1 - t) - L)
    inner = sorted(set(p for p in pts if lo < p < hi))
    return quad(f, [lo] + inner + [hi], maxdegree=10)
out = []
params = [(0.5,0.5),(0.5,5e5),(5e5,0.5),(2,3),(30,10),(0.1,0.1),(1e3,1e3),(1e3,7),(250,0.5),(5e4,5e4),(3,4e4),(1,1e5),(1e5,1e5+300)]
# moderate a next to a huge b (and the mirror), where the continued fraction is weakest near its switch point
params += [(100,5e5),(5e5,100),(10,1e5),(1e3,1e5),(30,3000),(3000,30)]
# tiny parameters: one side of I within 1e-20 of 1
params += [(1e-20,1),(1e-10,3),(3,1e-10),(1e-6,1e-6),(0.01,50)]
for a, b in params:
    A, B = mpf(a), mpf(b)
    m = A / (A + B); sd = (A * B / ((A + B) ** 2 * (A + B + 1))) ** 0.5
    xs = set()
    for k in (-8, -4, -2, -1, -0.3, 0.3, 1, 2, 4, 8):
        x = m + k * sd
        if 0 < x < 1: xs.add(float(x))
    xs |= {1e-6, 0.01, 0.5, 0.99}
    sw = (A + 1) / (A + B + 2)  # the switch point between the two sides
    xs |= {float(sw * (1 + d)) for d in (-1e-3, -1e-6, 0, 1e-6, 1e-3) if 0 < sw * (1 + d) < 1}
    for x in sorted(xs):
        X = mpf(x)
        # breakpoints: around the mode, and geometrically towards x on both sides
        slope = abs((A - 1) / X - (B - 1) / (1 - X)) + 1
        w = 1 / slope
        pts = [m + k * sd for k in (-40, -20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20, 40)]
        pts += [X - w * 2**k for k in range(0, 60)] + [X + w * 2**k for k in range(0, 60)]
        I = side(A, B, mpf(0), X, pts)
        J = side(B, A, mpf(0), 1 - X, [1 - p for p in pts])  # substitute s = 1 - t: the singular end is at 0
        if max(a, b) <= 1e3 and not (a == int(a) and b == int(b)):
            Ib = betainc(A, B, 0, X, regularized=True); Jb = betainc(A, B, X, 1, regularized=True)
            I, J = Ib, Jb
        if max(a, b) > 1e3:
            # the hypergeometric series (positive terms) at 40 digits, each side on its own
            L = lnB(A, B)
            pre = exp(A * log(X) + B * log(1 - X) - L)
            def ser(p, q, z):  # sum (p+q)_n/(p+1)_n z^n, positive terms
                t = mpf(1); acc = mpf(1); n = 0
                while True:
                    t *= (p + q + n) / (p + 1 + n) * z; acc += t; n += 1
                    if t < acc * mpf('1e-45'): return acc
            if X <= mpf('0.5'):
                Ih = pre / A * ser(A, B, X); Jh = 1 - Ih
            else:
                Jh = pre / B * ser(B, A, 1 - X); Ih = 1 - Jh
            I, J = Ih, Jh
            # a side formed as 1 − (the other) is only good to ~1e-40 absolute
            if I < mpf('1e-25'): I = None
            if J < mpf('1e-25'): J = None
        if a == int(a) and b == int(b) and a + b < 50000:
            n = int(a + b - 1); ai = int(a)
            Jx = fsum(binomial(n, j) * X**j * (1 - X)**(n - j) for j in range(0, ai))  # P(Bin(n, x) < a) = 1 - I
            Ix = fsum(binomial(n, j) * X**j * (1 - X)**(n - j) for j in range(ai, n + 1))
            I, J = Ix, Jx
        keep = lambda v: None if v is None or v < mpf('1e-290') else mp.nstr(v, 20)
        if keep(I) is None and keep(J) is None: continue
        out.append([a, b, x, keep(I), keep(J)])
print(json.dumps(out))
