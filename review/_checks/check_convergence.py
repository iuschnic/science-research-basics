"""Сходимость ШШ и точного метода по сетке в тонких и толстых полосах (поотносительной ошибке по полосам)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from exact_method import solve_exact  # noqa: E402
from opacity_table import OpacityTable, load_opacity_table  # noqa: E402
from schuster_schwarzschild import solve_schuster_schwarzschild  # noqa: E402

full = load_opacity_table(ROOT / "real_opacity_clean.txt")
args = (11000.0, 3000.0, 0.01, 2.0)
fr = (0.0, 0.5, 0.9)

# Полосы для проверки: тонкая, средняя, толстая, очень толстая (по tau_rad при k в м^-1)
pick = [0, 60, 125, 131, 133, 160]
sub = OpacityTable(full.bands[pick], full.temperatures_k, full.coefficients_m_inv[:, pick])

print("=== ШШ: flow_cells 1024 -> 16384, относительное изменение q в центре полосы")
a = solve_schuster_schwarzschild(sub, *args, radius_fractions=fr, flow_cells=1024)
b = solve_schuster_schwarzschild(sub, *args, radius_fractions=fr, flow_cells=16384)
for j, band in enumerate(pick):
    d = np.abs(a.div_flux[:, j, 4] - b.div_flux[:, j, 4]) / np.abs(b.div_flux[:, j, 4])
    print(f"  полоса #{band} tau_rad={a.tau_radial[j]:.3g}: {np.array2string(100*d, precision=2)} %")

print("=== Точный: ray_steps 192 -> 768, polar 20->32, azimuth 24->36")
a = solve_exact(sub, *args, radius_fractions=fr)
b = solve_exact(sub, *args, radius_fractions=fr, ray_steps=768, polar_order=32, azimuth_order=36)
for j, band in enumerate(pick):
    d = np.abs(a.div_flux[:, j, 4] - b.div_flux[:, j, 4]) / np.abs(b.div_flux[:, j, 4])
    print(f"  полоса #{band} tau_rad={a.tau_radial[j]:.3g}: {np.array2string(100*d, precision=2)} %")

ss = solve_schuster_schwarzschild(sub, *args, radius_fractions=fr, flow_cells=16384)
print("=== q_ШШ / q_точн (уточнённые сетки) в центре полосы, r/R = 0, 0.5, 0.9")
for j, band in enumerate(pick):
    print(f"  полоса #{band} tau_rad={b.tau_radial[j]:.3g}: {np.round(ss.div_flux[:, j, 4] / b.div_flux[:, j, 4], 3)}")

print("=== То же при k x100 (если таблица в см^-1)")
sub100 = OpacityTable(sub.bands, sub.temperatures_k, sub.coefficients_m_inv * 100)
e100 = solve_exact(sub100, *args, radius_fractions=fr, ray_steps=768, polar_order=32, azimuth_order=36)
s100 = solve_schuster_schwarzschild(sub100, *args, radius_fractions=fr, flow_cells=16384)
for j, band in enumerate(pick):
    print(f"  полоса #{band} tau_rad={e100.tau_radial[j]:.3g}: {np.round(s100.div_flux[:, j, 4] / e100.div_flux[:, j, 4], 3)}")
