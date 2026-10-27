# Reference grid for the regularized incomplete gamma P(a, x), Q(a, x) (tests/fixtures/gamma-mpmath.json), 40 digits.
# mpmath's gammainc up to a = 1e5; for a ≥ 1e6 (where it gives up) the positive series for P left of the mean and the
# Legendre continued fraction for Q right of it, the other side as 1 − that one (kept only when it is ≥ 1e-25).
# Run: PYTHONPATH=<site-packages with mpmath 1.3> python3 gamma_grid.py > gamma-mpmath.json
from mpmath import mp, mpf, gammainc, inf, exp, log, loggamma
import json
mp.dps = 40

def big_shape(A, X):
    pre = exp(-X + A * log(X) - loggamma(A))
    if X < A:
        t = mpf(1) / A; s = t; n = 1
        while t > s * mpf('1e-45'):
            t *= X / (A + n); s += t; n += 1
        P = pre * s
        return P, 1 - P
    # Q = pre / F, F = b0 + a1/(b1 + a2/(b2 + …)) with bj = x + 1 − a + 2j, aj = −j(j − a): Legendre's continued
    # fraction (DLMF 8.9.2) in its even contraction, by modified Lentz (Lentz 1976; Thompson & Barnett 1986)
    tiny = mpf('1e-300'); f = X + 1 - A; C = f; D = mpf(0); j = 1
    while True:
        aj = -j * (j - A); bj = X + 1 - A + 2 * j
        D = bj + aj * D; D = D if abs(D) > tiny else tiny
        C = bj + aj / C; C = C if abs(C) > tiny else tiny
        D = 1 / D; delta = C * D; f *= delta; j += 1
        if abs(delta - 1) < mpf('1e-45'): break
    Q = pre / f
    return 1 - Q, Q

out = []
for a in (1e-310, 1e-300, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1, 2.5, 10, 30.5, 100, 1e3, 1e4, 1e5, 1e6, 1e7):
    A = mpf(a); sd = A ** 0.5
    xs = {1e-300, 1e-6, 0.01, 0.3, 1, 1.5, 10, 200}
    for k in (-8, -4, -2, -1, -0.3, 0.3, 1, 2, 4, 8, 16):
        x = a + k * sd
        if x > 0: xs.add(float(x))
    for x in sorted(xs):
        X = mpf(x)
        if a >= 1e6:
            P, Q = big_shape(A, X)
            if (X < A and Q < mpf('1e-25')) or (X >= A and P < mpf('1e-25')): continue  # complement not trustworthy
        else:
            P = gammainc(A, 0, X, regularized=True); Q = gammainc(A, X, inf, regularized=True)
        keep = lambda v: None if v < mpf('1e-300') else mp.nstr(v, 20)
        if keep(P) is None and keep(Q) is None: continue
        out.append([a, x, keep(P), keep(Q)])
print(json.dumps(out))
