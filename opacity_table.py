"""Чтение таблицы линейных коэффициентов поглощения k(T, ν*)."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OpacityTable:
    bands: NDArray[np.float64]             # [полоса, левая/правая граница ν*]
    temperatures_k: NDArray[np.float64]    # [строка таблицы], К
    coefficients_m_inv: NDArray[np.float64]  # [температура, полоса], м⁻¹

    def interpolate(self, temperature_k: NDArray, band: int) -> NDArray:
        """Внутри полосы k зависит от T через линейную интерполяцию по строкам."""
        return np.interp(
            temperature_k, self.temperatures_k, self.coefficients_m_inv[:, band]
        )


def load_opacity_table(path: Path) -> OpacityTable:
    """Заголовок с интервалами, затем пары строк «температура / ряд k»."""
    lines = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    if len(lines) < 3 or (len(lines) - 1) % 2:
        raise ValueError(f"{path.name}: после заголовка нужны пары строк T и k.")

    number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    interval = rf"\s*({number})\s*[-–—]\s*({number})\s*"
    bands = []
    for field in lines[0].split(","):
        match = re.fullmatch(interval, field)
        if match is None:
            raise ValueError(f"{path.name}: неверный интервал ν* {field!r}.")
        bands.append(tuple(map(float, match.groups())))

    temperatures = []
    coefficients = []
    for i in range(1, len(lines), 2):
        # Допускаются, например, 2000, 2000K, 2000 K и 2000 К.
        temperature = lines[i].rstrip("KkКк").strip()
        temperatures.append(float(temperature))
        row = [float(value) for value in re.split(r"[\s,]+", lines[i + 1])]
        if len(row) != len(bands):
            raise ValueError(f"{path.name}: при T={temperature} нужен k для каждой полосы.")
        coefficients.append(row)

    band_array = np.array(bands)
    temperature_array = np.array(temperatures)
    coefficient_array = np.array(coefficients)
    if not all(np.isfinite(values).all() for values in (
        band_array, temperature_array, coefficient_array
    )):
        raise ValueError(f"{path.name}: таблица должна содержать конечные числа.")
    if np.any(band_array[:, 0] >= band_array[:, 1]) or np.any(
        band_array[1:, 0] < band_array[:-1, 1]
    ):
        raise ValueError(f"{path.name}: интервалы должны идти по порядку без пересечений.")
    if np.any(temperature_array <= 0) or np.any(np.diff(temperature_array) <= 0):
        raise ValueError(f"{path.name}: температуры должны быть положительными и расти.")
    if np.any(coefficient_array < 0):
        raise ValueError(f"{path.name}: коэффициенты поглощения не могут быть отрицательными.")
    return OpacityTable(band_array, temperature_array, coefficient_array)
