"""Чтение таблицы линейных коэффициентов поглощения k(T, ν*) и интерполяция по T.

Поддерживаются два формата файла:

* «чистый» (`real_opacity_clean.txt`, `demo_opacity.txt`): первая строка —
  интервалы ν* через запятую, затем пары строк «температура / ряд k»;
* исходный формат преподавателя (`real_opacity.txt`, кодировка cp1251):
  текстовые подписи, число интервалов nint, nint+1 узлов сетки ν*,
  число температур, список температур и ntemp блоков по nint значений k.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

INTERPOLATIONS = ("linear", "log")
EXTRAPOLATIONS = ("error", "clamp", "log-linear")


@dataclass(frozen=True)
class OpacityTable:
    bands: NDArray[np.float64]               # [полоса, левая/правая граница ν*]
    temperatures_k: NDArray[np.float64]      # [строка таблицы], К
    coefficients_m_inv: NDArray[np.float64]  # [температура, полоса], м⁻¹ (после перевода единиц)
    # "linear" — линейно по k; "log" — линейно по ln k (k меняется на порядки).
    interpolation: str = "linear"
    # Вне диапазона таблицы: "error" — запрет, "clamp" — значение крайней
    # строки, "log-linear" — продолжение ln k по двум крайним строкам.
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation должна быть одной из {INTERPOLATIONS}.")
        if self.extrapolation not in EXTRAPOLATIONS:
            raise ValueError(f"extrapolation должна быть одной из {EXTRAPOLATIONS}.")
        if self.interpolation == "log" or self.extrapolation == "log-linear":
            if np.any(self.coefficients_m_inv <= 0):
                raise ValueError("Логарифмическая интерполяция требует k > 0.")

    def interpolate_all(self, temperature_k: NDArray) -> NDArray:
        """k для массива температур во всех полосах, оси [температура, полоса]."""
        return self._evaluate(temperature_k, slice(None))

    def _evaluate(self, temperature_k: NDArray, columns: slice) -> NDArray:
        T = np.atleast_1d(np.asarray(temperature_k, dtype=float))
        Ts = self.temperatures_k
        values = self.coefficients_m_inv[:, columns]
        use_log = self.interpolation == "log"
        data = np.log(values) if use_log else values

        # Положение между строками и вес правой строки (с обрезкой по краям).
        right = np.clip(np.searchsorted(Ts, T), 1, len(Ts) - 1)
        left = right - 1
        weight = np.clip((T - Ts[left]) / (Ts[right] - Ts[left]), 0.0, 1.0)
        result = data[left] * (1 - weight)[:, None] + data[right] * weight[:, None]
        if use_log:
            result = np.exp(result)

        if self.extrapolation == "log-linear":
            logs = np.log(values)
            for mask, i, j in ((T < Ts[0], 0, 1), (T > Ts[-1], -1, -2)):
                if mask.any():
                    slope = (logs[i] - logs[j]) / (Ts[i] - Ts[j])
                    result[mask] = np.exp(logs[i] + slope * (T[mask, None] - Ts[i]))
        return result

    def interpolate(self, temperature_k: NDArray, band: int) -> NDArray:
        """k(T) в одной полосе; форма результата совпадает с формой temperature_k."""
        T = np.asarray(temperature_k, dtype=float)
        return self._evaluate(T.ravel(), slice(band, band + 1))[:, 0].reshape(T.shape)

    def covers(self, t_min: float, t_max: float) -> bool:
        """Можно ли получить k на отрезке температур без запрещённой экстраполяции."""
        inside = self.temperatures_k[0] <= t_min and t_max <= self.temperatures_k[-1]
        return inside or self.extrapolation != "error"


def _validated(path: Path, bands, temperatures, coefficients, unit_scale, **options) -> OpacityTable:
    band_array = np.array(bands, dtype=float)
    temperature_array = np.array(temperatures, dtype=float)
    coefficient_array = np.array(coefficients, dtype=float) * unit_scale
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
    return OpacityTable(band_array, temperature_array, coefficient_array, **options)


def _parse_clean(path: Path, lines: list[str], unit_scale: float, **options) -> OpacityTable:
    """Заголовок с интервалами, затем пары строк «температура / ряд k»."""
    lines = [line.split("#", 1)[0].strip() for line in lines]
    lines = [line for line in lines if line]
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
    return _validated(path, bands, temperatures, coefficients, unit_scale, **options)


def _parse_teacher(path: Path, lines: list[str], unit_scale: float, **options) -> OpacityTable:
    """Формат преподавателя: группы чисел между текстовыми строками-подписями."""
    groups: list[list[float]] = []
    previous_numeric = False
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            previous_numeric = False  # Строка-подпись начинает новую группу.
            continue
        if previous_numeric:
            groups[-1].extend(values)
        else:
            groups.append(values)
        previous_numeric = True

    if len(groups) != 5:
        raise ValueError(f"{path.name}: ожидалось 5 числовых групп (nint, сетка, ntemp, T, k), "
                         f"найдено {len(groups)}.")
    (nint,), grid, (ntemp,), temperatures, flat = groups
    nint, ntemp = int(nint), int(ntemp)
    if len(grid) != nint + 1:
        raise ValueError(f"{path.name}: для nint={nint} нужно {nint + 1} узлов ν*, найдено {len(grid)}.")
    if len(temperatures) != ntemp:
        raise ValueError(f"{path.name}: ожидалось {ntemp} температур, найдено {len(temperatures)}.")
    if len(flat) != nint * ntemp:
        raise ValueError(f"{path.name}: ожидалось {nint * ntemp} значений k, найдено {len(flat)}.")
    bands = np.column_stack((grid[:-1], grid[1:]))
    coefficients = np.reshape(flat, (ntemp, nint))
    return _validated(path, bands, temperatures, coefficients, unit_scale, **options)


def load_opacity_table(
    path: Path, *, unit_scale: float = 1.0,
    interpolation: str = "linear", extrapolation: str = "error",
) -> OpacityTable:
    """Читает таблицу любого из двух форматов.

    unit_scale переводит значения файла в м⁻¹: 1 для м⁻¹, 100 для см⁻¹.
    """
    if not unit_scale > 0:
        raise ValueError("unit_scale должен быть положительным.")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    lines = text.splitlines()
    first = next((line.strip() for line in lines if line.strip()), "")
    options = dict(interpolation=interpolation, extrapolation=extrapolation)
    if re.match(r"^[\d.]+\s*[-–—]", first):
        return _parse_clean(path, lines, unit_scale, **options)
    return _parse_teacher(path, lines, unit_scale, **options)
