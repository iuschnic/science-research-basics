"""Запуск точного и двухпоточного расчётов и сохранение спектров."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Для сохранения PNG окно графика не требуется.

from exact_method import solve_exact
from opacity_table import load_opacity_table
from plots import save_comparison_plots
from schuster_schwarzschild import solve_schuster_schwarzschild

PROJECT_DIR = Path(__file__).resolve().parent
TABLE_PATH = PROJECT_DIR / "real_opacity_clean.txt"
OUTPUT_DIR = PROJECT_DIR / "output"

T0 = 11_000.0    # К, температура на оси цилиндра
TW = 3_000.0     # К, температура среды на боковой границе
R = 0.01        # м, радиус цилиндра
N = 2.0         # Показатель степени в заданном T(r)
FREQUENCY_SCALE_HZ = 1e15  # Рабочий масштаб ν = ν* × FREQUENCY_SCALE_HZ.
RADIUS_FRACTIONS = (0.0, 0.5, 0.9, 1.0)


def main() -> None:
    table = load_opacity_table(TABLE_PATH)
    options = dict(
        frequency_scale_hz=FREQUENCY_SCALE_HZ,
        radius_fractions=RADIUS_FRACTIONS,
    )
    print("Расчёт точного метода…", flush=True)
    exact = solve_exact(table, T0, TW, R, N, **options)
    print("Расчёт Шустера — Шварцшильда и построение графиков…", flush=True)
    approx = solve_schuster_schwarzschild(table, T0, TW, R, N, **options)
    save_comparison_plots(exact, approx, OUTPUT_DIR, R, FREQUENCY_SCALE_HZ)
    print(f"Графики сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
