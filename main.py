"""Запуск точного и двухпоточного расчётов, сохранение графиков и таблиц."""

import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Для сохранения PNG окно графика не требуется.

from exact_method import solve_exact
from opacity_table import load_opacity_table
from plots import save_comparison_plots
from results_io import save_results_txt
from schuster_schwarzschild import solve_schuster_schwarzschild

PROJECT_DIR = Path(__file__).resolve().parent
# Исходный файл преподавателя (cp1251) читается напрямую.
TABLE_PATH = PROJECT_DIR / "real_opacity.txt"
OUTPUT_DIR = PROJECT_DIR / "output"

# ---- Параметры варианта --------------------------------------------------
# TODO: T0, N, R сверить с индивидуальным заданием (на листе: «T0 и n — задано»).
T0 = 11_000.0    # К, температура на оси цилиндра
TW = 1_800.0     # К, температура у боковой границы (по рукописному заданию)
R = 0.01         # м, радиус цилиндра
N = 2.0          # Показатель степени в T(r) = T0 + (TW − T0)(r/R)^N

# ---- Таблица коэффициентов поглощения -----------------------------------
# TODO: уточнить у преподавателя единицы k в файле. 1 — м⁻¹, 100 — см⁻¹.
K_UNIT_SCALE = 1.0
# "linear" — линейно по k (как раньше), "log" — линейно по ln k.
K_INTERPOLATION = "linear"
# Таблица начинается с 2000 К, а TW = 1800 К. Нужна экстраполяция:
# "clamp" — k(T < 2000) = k(2000 К), "log-linear" — продолжение ln k.
# TODO: согласовать выбор с преподавателем ("error" — запретить выход за таблицу).
K_EXTRAPOLATION = "clamp"

FREQUENCY_SCALE_HZ = 1e15  # ν = ν* × FREQUENCY_SCALE_HZ
RADIUS_FRACTIONS = tuple(np.round(np.linspace(0.0, 1.0, 21), 10))  # для зависимостей от r/R
SPECTRUM_FRACTIONS = (0.0, 0.5, 0.9, 1.0)  # для графиков спектров
WORKERS = None  # процессы для точного метода; None — по числу ядер


def main() -> None:
    table = load_opacity_table(
        TABLE_PATH, unit_scale=K_UNIT_SCALE,
        interpolation=K_INTERPOLATION, extrapolation=K_EXTRAPOLATION,
    )
    options = dict(frequency_scale_hz=FREQUENCY_SCALE_HZ, radius_fractions=RADIUS_FRACTIONS)
    description = (
        f"T0={T0:g} K, Tw={TW:g} K, R={R:g} m, n={N:g}; таблица {TABLE_PATH.name}, "
        f"k×{K_UNIT_SCALE:g} -> м^-1, интерполяция {K_INTERPOLATION}, экстраполяция {K_EXTRAPOLATION}"
    )
    print(description)

    start = time.perf_counter()
    print(f"Точный метод ({len(RADIUS_FRACTIONS)} точек по радиусу)…", flush=True)
    exact = solve_exact(table, T0, TW, R, N, workers=WORKERS, **options)
    print(f"  готово за {time.perf_counter() - start:.0f} с", flush=True)

    print("Шустер — Шварцшильд…", flush=True)
    approx = solve_schuster_schwarzschild(table, T0, TW, R, N, **options)

    save_results_txt(exact, approx, OUTPUT_DIR, R, FREQUENCY_SCALE_HZ, description)
    save_comparison_plots(exact, approx, OUTPUT_DIR, R, FREQUENCY_SCALE_HZ, SPECTRUM_FRACTIONS)
    print(f"Графики и таблицы сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
