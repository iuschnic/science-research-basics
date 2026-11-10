"""Сходимость нового ШШ (дефицит + сгущение) по сетке во всех полосах, k в м^-1 и ×100."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from opacity_table import load_opacity_table  # noqa: E402
from schuster_schwarzschild import solve_schuster_schwarzschild  # noqa: E402

fr = (0.0, 0.5, 0.9, 1.0)
for scale in (1.0, 100.0):
    table = load_opacity_table(ROOT / "real_opacity.txt", unit_scale=scale, extrapolation="clamp")
    args = (table, 11000.0, 1800.0, 0.01, 2.0)
    a = solve_schuster_schwarzschild(*args, radius_fractions=fr)
    b = solve_schuster_schwarzschild(*args, radius_fractions=fr, flow_cells=4096, refine_cells=960)
    rel = np.abs(a.div_flux - b.div_flux) / np.abs(b.div_flux).max(axis=-1, keepdims=True)
    worst = rel.max(axis=(0, 2))
    i = np.argmax(worst)
    print(f"k×{scale:g}: макс. изменение q по полосе (норма — max|q| в полосе на радиусе): "
          f"{100*worst.max():.3f}% (полоса #{i}, tau={a.tau_radial[i]:.3g}); "
          f"медиана {100*np.median(worst):.2e}%")
    for band in np.argsort(worst)[-4:]:
        j = np.argmax(rel[:, band].max(axis=1))
        print(f"   полоса #{band} tau={a.tau_radial[band]:.3g} r/R={fr[j]}: "
              f"q(1024)={a.div_flux[j, band, 4]:.4e}, q(fine)={b.div_flux[j, band, 4]:.4e}, "
              f"q_fine/max|q_fine по радиусам| = {b.div_flux[j, band, 4]/np.abs(b.div_flux[:, band, 4]).max():.2e}")
    c = solve_schuster_schwarzschild(*args, radius_fractions=fr, flow_cells=16384, refine_cells=2000)
    rel2 = np.abs(b.div_flux - c.div_flux) / np.abs(c.div_flux).max(axis=-1, keepdims=True)
    print(f"   4096/960 -> 16384/2000: макс. {100*rel2.max():.3f}%")
