"""Выбросы в таблице, запуск обоих методов, баланс энергии, тесты на однородном цилиндре."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from exact_method import solve_exact  # noqa: E402
from opacity_table import OpacityTable, load_opacity_table  # noqa: E402
from schuster_schwarzschild import solve_schuster_schwarzschild  # noqa: E402

table = load_opacity_table(ROOT / "real_opacity_clean.txt")
kc, Ts, bands = table.coefficients_m_inv, table.temperatures_k, table.bands

print("=== 1. Выбросы: значение отличается от геом. среднего соседей по T более чем в 4 раза,")
print("    а соседи по T между собой отличаются менее чем в 2 раза")
lk = np.log10(kc)
for i in range(1, len(Ts) - 1):
    mid = (lk[i - 1] + lk[i + 1]) / 2
    dev = lk[i] - mid
    smooth = np.abs(lk[i + 1] - lk[i - 1]) < np.log10(2)
    for b in np.where((np.abs(dev) > np.log10(4)) & smooth)[0]:
        print(f"  T={Ts[i]:.0f} K, полоса #{b} {bands[b][0]:.5f}-{bands[b][1]:.5f}: k={kc[i, b]:.3g}, "
              f"соседи {kc[i-1, b]:.3g} / {kc[i+1, b]:.3g}")

print("=== 2. Tw = 1800 K (из рукописных заметок)")
try:
    solve_schuster_schwarzschild(table, 11000.0, 1800.0, 0.01, 2.0)
except ValueError as e:
    print("  ValueError:", e)

print("=== 3. Однородный цилиндр, одна полоса, сравнение с аналитикой ШШ и точной формулой на оси")
for kval in (1.0, 100.0, 1e4):
    t1 = OpacityTable(np.array([[0.3, 0.4]]), np.array([1000.0, 20000.0]), np.array([[kval], [kval]]))
    R = 0.01
    ss = solve_schuster_schwarzschild(t1, 8000, 8000, R, 2.0, radius_fractions=(0.0, 0.5, 1.0))
    ex = solve_exact(t1, 8000, 8000, R, 2.0, radius_fractions=(0.0, 0.5, 1.0), ray_steps=64)
    from transfer_common import planck
    nu = ss.nu_star[0] * 1e15
    S = np.pi * planck(nu, 8000.0)
    r = np.array([0.0, 0.5, 1.0]) * R
    Fm = S * (1 - np.exp(-2 * kval * (R - r)))[:, None]
    frac = np.where(r > 0, (1 - np.exp(-4 * kval * r)) / np.where(r > 0, 4 * kval * r, 1), 1.0)
    Fp = S * (1 - np.exp(-2 * kval * (R - r)) * frac)[:, None]
    err = max(np.abs(ss.flux_minus[:, 0] - Fm).max(), np.abs(ss.flux_plus[:, 0] - Fp).max()) / S.max()
    q_ratio = ss.div_flux[:, 0, 4] / ex.div_flux[:, 0, 4]
    print(f"  k={kval:g} м^-1 (kR={kval*R:g}): ШШ vs аналитика ШШ max отн. ошибка {err:.2e}; "
          f"q_ШШ/q_точн при r/R=0,0.5,1: {np.round(q_ratio, 3)}")

print("=== 4. Основной расчёт (параметры main.py)")
opts = dict(frequency_scale_hz=1e15, radius_fractions=tuple(np.linspace(0, 1, 21)))
t = time.time()
ex = solve_exact(table, 11000.0, 3000.0, 0.01, 2.0, **opts)
print(f"  точный: {time.time()-t:.0f} с")
t = time.time()
ss = solve_schuster_schwarzschild(table, 11000.0, 3000.0, 0.01, 2.0, **opts)
print(f"  ШШ: {time.time()-t:.1f} с")
np.savez(OUT / "run_main.npz", nu=ex.nu_star, r=ex.radii_m, tau=ex.tau_radial,
         q_ex=ex.div_flux, q_ss=ss.div_flux, fp_ex=ex.flux_plus, fm_ex=ex.flux_minus,
         fp_ss=ss.flux_plus, fm_ss=ss.flux_minus)

# Интегралы по частоте (трапеции внутри полос), Вт/м^3
nu_hz = ex.nu_star * 1e15


def integ(q):
    return np.trapezoid(q, nu_hz, axis=-1).sum(axis=-1)


Q_ex, Q_ss = integ(ex.div_flux), integ(ss.div_flux)
print("  r/R   Q_точн, Вт/м3     Q_ШШ, Вт/м3    отн. расх., %")
for x, a, b in zip(ex.radii_m / 0.01, Q_ex, Q_ss):
    print(f"  {x:4.2f}  {a: .4e}  {b: .4e}  {100*(b-a)/abs(a): .1f}")

# Баланс энергии: ∫ q 2πr dr = 2πR F(R), для интегрированных по частоте величин
r = ex.radii_m
for name, res, Q in (("точный", ex, Q_ex), ("ШШ", ss, Q_ss)):
    lhs = np.trapezoid(Q * 2 * np.pi * r, r)
    FR = integ(res.flux_plus - res.flux_minus)[-1]
    print(f"  баланс {name}: ∫q·2πr dr = {lhs:.4e}, 2πR·F(R) = {2*np.pi*0.01*FR:.4e} Вт/м "
          f"(расх. {100*(lhs/(2*np.pi*0.01*FR)-1):.1f}% — грубая сетка 21 точка)")

# Ошибка по спектру: доля частотных точек с ошибкой > 10 %, нормированная на max|q|
for idx in (0, 10, 18, 20):
    a, b = ex.div_flux[idx], ss.div_flux[idx]
    norm = np.abs(a).max()
    rel_max = np.abs(b - a) / norm
    sign_flip = np.mean(np.sign(a) != np.sign(b))
    print(f"  r/R={r[idx]/0.01:.2f}: max|Δq|/max|q_точн| = {100*rel_max.max():.1f}%, "
          f"доля точек со сменой знака q = {100*sign_flip:.1f}%")

# Ошибка по полосам в зависимости от tau
tau = ex.tau_radial
band_err = np.abs(ss.div_flux[0] - ex.div_flux[0]).max(axis=1) / np.abs(ex.div_flux[0]).max(axis=1)
for lo, hi in ((0, 0.1), (0.1, 1), (1, 10), (10, 1e9)):
    m = (tau >= lo) & (tau < hi)
    if m.any():
        print(f"  ось, полосы tau∈[{lo},{hi}): {m.sum()} шт, медиана отн. ошибки q {100*np.median(band_err[m]):.1f}%,"
              f" max {100*band_err[m].max():.1f}%")
