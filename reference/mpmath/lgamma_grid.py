# Reference sweep for lgamma (tests/fixtures/lgamma-mpmath.json): mpmath loggamma at 40 digits, x from 1e-310 to
# 1e305 (three mantissas per 7 decades), the region [0.25, 1e5] and a few negative non-integers (ln|Γ|).
# Run: PYTHONPATH=<site-packages with mpmath 1.3> python3 lgamma_grid.py > lgamma-mpmath.json
from mpmath import mp, mpf, loggamma, log, gamma
import json, math
mp.dps = 40
xs = []
for e in range(-310, 306, 7):
    for m in (1.0, 2.7, 6.3):
        x = m * 10.0 ** e if e > -308 else m * 10.0 ** (e + 20) * 1e-20
        if x > 0 and math.isfinite(x): xs.append(x)
xs += [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3, 4.5, 7.5, 10, 14.9, 15, 15.1, 20, 50, 100, 1e3, 1e4, 1e5]
xs += [-0.5, -1.5, -2.25, -10.5, -100.25]
out = []
for x in sorted(set(xs)):
    X = mpf(x)
    v = loggamma(X) if x > 0 else log(abs(gamma(X)))
    out.append([x, mp.nstr(mp.re(v), 25)])
print(json.dumps(out))
