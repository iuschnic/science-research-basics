"""Чувствительность Q(r) и точности ШШ к нерешённым настройкам таблицы k."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from exact_method import solve_exact  # noqa: E402
from opacity_table import load_opacity_table  # noqa: E402
from schuster_schwarzschild import solve_schuster_schwarzschild  # noqa: E402
from transfer_common import integrate_over_frequency  # noqa: E402

FR = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
CASES = {
    "базовый (м^-1, linear, clamp)": dict(unit_scale=1, interpolation="linear", extrapolation="clamp"),
    "log-интерполяция": dict(unit_scale=1, interpolation="log", extrapolation="clamp"),
    "log-linear экстраполяция": dict(unit_scale=1, interpolation="linear", extrapolation="log-linear"),
    "k в см^-1 (×100)": dict(unit_scale=100, interpolation="linear", extrapolation="clamp"),
}

if __name__ == "__main__":
    print("r/R:", FR)
    for name, opts in CASES.items():
        table = load_opacity_table(ROOT / "real_opacity.txt", **opts)
        args = (table, 11000.0, 1800.0, 0.01, 2.0)
        ex = solve_exact(*args, radius_fractions=FR, workers=None)
        ss = solve_schuster_schwarzschild(*args, radius_fractions=FR)
        qe = integrate_over_frequency(ex.div_flux, ex.nu_star, 1e15)
        qs = integrate_over_frequency(ss.div_flux, ss.nu_star, 1e15)
        print(f"\n== {name}")
        print("  Q_точн, Вт/м3:", np.array2string(qe, precision=3))
        print("  Q_ШШ/Q_точн:  ", np.array2string(qs / qe, precision=3))
        tau = ex.tau_radial
        print(f"  полос tau<0.1: {(tau<0.1).sum()}, 0.1..10: {((tau>=0.1)&(tau<=10)).sum()}, >10: {(tau>10).sum()}")
        sys.stdout.flush()
