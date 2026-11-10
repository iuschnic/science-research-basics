"""Числа для статьи из output/txt последнего запуска main.py."""
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
txt = ROOT / "output" / "txt"
tau = np.loadtxt(txt / "tau_radial.txt")
qe = np.loadtxt(txt / "div_flux_exact.txt")
qs = np.loadtxt(txt / "div_flux_ss.txt")
fe = np.loadtxt(txt / "flux_exact.txt")
fs = np.loadtxt(txt / "flux_ss.txt")
integ = np.loadtxt(txt / "integrated.txt")
t = tau[:, 3]
nb = len(t)
print("полос:", nb, "tau min/median/max:", t.min(), np.median(t), t.max())
print("tau<0.1:", (t < 0.1).sum(), "0.1..1:", ((t >= 0.1) & (t < 1)).sum(),
      "1..10:", ((t >= 1) & (t < 10)).sum(), ">=10:", (t >= 10).sum())
nu = qe[:, 1].reshape(nb, -1)
fr = integ[:, 0]
cols = {0.0: 0, 0.5: 10, 0.9: 18, 1.0: 20}
for x, c in cols.items():
    a = qe[:, 2 + c].reshape(nb, -1)[:, 4]
    b = qs[:, 2 + c].reshape(nb, -1)[:, 4]
    ratio = b / a
    print(f"r/R={x}:")
    for lo, hi in ((0, 0.1), (0.1, 1), (1, 10), (10, 1e9)):
        m = (t >= lo) & (t < hi)
        print(f"   tau[{lo},{hi}): n={m.sum()}, отношение q_ШШ/q_точн: медиана {np.median(ratio[m]):.3f}, "
              f"min {ratio[m].min():.3f}, max {ratio[m].max():.3f}")
# вклад классов полос в Q на оси
w = np.trapezoid(qe[:, 2].reshape(nb, -1), nu * 1e15, axis=1)
ws = np.trapezoid(qs[:, 2].reshape(nb, -1), nu * 1e15, axis=1)
for lo, hi in ((0, 0.1), (0.1, 1), (1, 10), (10, 1e9)):
    m = (t >= lo) & (t < hi)
    print(f"вклад в Q(0) полос tau[{lo},{hi}): точн {100*w[m].sum()/w.sum():.1f}%, ШШ/точн по классу {ws[m].sum()/w[m].sum():.3f}")
# выходящий поток с поверхности и баланс
Fe = np.trapezoid(fe[:, -1].reshape(nb, -1), nu * 1e15, axis=1).sum()
Fs = np.trapezoid(fs[:, -1].reshape(nb, -1), nu * 1e15, axis=1).sum()
R = 0.01
r = fr * R
print(f"F(R): точн {Fe:.4e}, ШШ {Fs:.4e} Вт/м2, отношение {Fs/Fe:.4f}")
for name, Q, F in (("точн", integ[:, 1], Fe), ("ШШ", integ[:, 2], Fs)):
    lhs = np.trapezoid(Q * 2 * np.pi * r, r)
    print(f"баланс {name}: {lhs:.4e} vs {2*np.pi*R*F:.4e} Вт/м ({100*(lhs/(2*np.pi*R*F)-1):+.2f}%)")
print("Q:", integ[[0, 10, 20], 1], "смена знака между", fr[np.where(np.diff(np.sign(integ[:, 1])))[0]])
print("T(r/R=0.5, 0.9):", 11000 - 9200 * 0.25, 11000 - 9200 * 0.81)

figs = ROOT / "review" / "figures"
figs.mkdir(exist_ok=True)
out = ROOT / "output"
for src, dst in ((out / "exact" / "tau_radial.png", "fig1_tau.png"),
                 (out / "comparison" / "values" / "div_flux_01_r_over_R_0.png", "fig2_spectrum_axis.png"),
                 (out / "comparison" / "integrated_div_flux_vs_radius.png", "fig3_integrated.png"),
                 (out / "comparison" / "ratio_vs_tau.png", "fig4_ratio_vs_tau.png")):
    shutil.copyfile(src, figs / dst)
    print("copied", dst)
