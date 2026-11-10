"""Проверки входных данных: сверка исходного файла преподавателя с real_opacity_clean.txt."""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from opacity_table import load_opacity_table  # noqa: E402

raw = (ROOT / "real_opacity.txt").read_text(encoding="cp1251").splitlines()
num = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+")


def numbers(lines):
    out = []
    for line in lines:
        out += [float(x) for x in num.findall(line)]
    return out


nint = int(raw[1])
grid = numbers(raw[3:25])
ntemp = int(raw[26])
temps = numbers(raw[28:29])
blocks = numbers(raw[30:])
print("nint =", nint, " grid points =", len(grid), " ntemp =", ntemp, " temps =", temps)
print("k values total =", len(blocks), " per T =", len(blocks) / ntemp)
kraw = np.array(blocks).reshape(ntemp, -1)

table = load_opacity_table(ROOT / "real_opacity_clean.txt")
print("clean: bands =", len(table.bands), " first band", table.bands[0], " last", table.bands[-1])
kc = table.coefficients_m_inv
print("clean shape", kc.shape)
# какой столбец исходника выпал
print("clean k == raw k:", np.array_equal(kraw, kc))
edges = np.r_[table.bands[:, 0], table.bands[-1, 1]]
print("clean bands == raw grid:", np.allclose(edges, grid))
print("raw temps == clean temps:", np.array_equal(temps, table.temperatures_k))

# аномалии
last = kc[:, -1]
print("последняя полоса 2.5267-3.0 по T:", dict(zip(table.temperatures_k.astype(int), last)))
ratio = kc[1:] / kc[:-1]
i, j = np.unravel_index(np.argmax(ratio), ratio.shape)
print(f"макс. скачок k между соседними T: x{ratio[i, j]:.3g} в полосе {table.bands[j]} при {table.temperatures_k[i]}->{table.temperatures_k[i+1]}")
print("k min/max:", kc.min(), kc.max())

# оптическая толщина для профиля из main при разных предположениях о единицах
T0, TW, R, N = 11000.0, 3000.0, 0.01, 2.0
rho = (np.arange(4096) + 0.5) / 4096 * R
T = T0 + (TW - T0) * (rho / R) ** N
tau = np.array([np.interp(T, table.temperatures_k, kc[:, b]).sum() * R / 4096 for b in range(kc.shape[1])])
for label, mult in (("k в м^-1 (как в коде)", 1), ("k в см^-1 (x100)", 100)):
    t = tau * mult
    print(f"{label}: tau min={t.min():.3g} median={np.median(t):.3g} max={t.max():.3g}; "
          f"полос tau<0.1: {(t < 0.1).sum()}, 0.1..10: {((t >= 0.1) & (t <= 10)).sum()}, >10: {(t > 10).sum()}")

# линейная vs логарифмическая интерполяция в середине интервала
Tm = np.array([7500.0])
lin = np.array([np.interp(Tm, table.temperatures_k, kc[:, b])[0] for b in range(kc.shape[1])])
logi = np.array([np.exp(np.interp(Tm, table.temperatures_k, np.log(kc[:, b])))[0] for b in range(kc.shape[1])])
rel = lin / logi
print(f"T=7500K: линейная/лог-интерполяция k: медиана {np.median(rel):.3g}, макс {rel.max():.3g}")
Tm = np.array([2500.0])
lin = np.array([np.interp(Tm, table.temperatures_k, kc[:, b])[0] for b in range(kc.shape[1])])
logi = np.array([np.exp(np.interp(Tm, table.temperatures_k, np.log(kc[:, b])))[0] for b in range(kc.shape[1])])
rel = lin / logi
print(f"T=2500K: линейная/лог-интерполяция k: медиана {np.median(rel):.3g}, макс {rel.max():.3g}")
