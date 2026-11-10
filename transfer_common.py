"""Входная сетка и физические величины, общие для двух методов."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opacity_table import OpacityTable

PLANCK_CONSTANT = 6.62607015e-34  # Дж·с
BOLTZMANN_CONSTANT = 1.380649e-23  # Дж/К
LIGHT_SPEED = 299_792_458.0       # м/с, показатель преломления считаем равным 1


@dataclass(frozen=True)
class TransferResult:
    nu_star: NDArray[np.float64]      # [полоса, отсчёт внутри полосы]
    tau_radial: NDArray[np.float64]   # [полоса], безразмерная
    radii_m: NDArray[np.float64]      # [точка наблюдения]
    # Далее оси [точка наблюдения, полоса, отсчёт внутри полосы].
    flux_plus: NDArray[np.float64]    # Вт/(м²·Гц), наружу
    flux_minus: NDArray[np.float64]   # Вт/(м²·Гц), к оси, записан положительным
    div_flux: NDArray[np.float64]     # Вт/(м³·Гц), знак плюс означает охлаждение

    @property
    def flux(self) -> NDArray[np.float64]:
        return self.flux_plus - self.flux_minus


def integrate_over_frequency(spectrum: NDArray, nu_star: NDArray, frequency_scale_hz: float) -> NDArray:
    """∫ spectrum dν по всем полосам (трапеции внутри каждой полосы), ν в Гц.

    Для div_flux даёт Q(r) в Вт/м³ — величину, входящую в уравнение энергии.
    Последняя ось spectrum — отсчёты внутри полосы, предпоследняя — полосы.
    """
    return np.trapezoid(spectrum, nu_star * frequency_scale_hz, axis=-1).sum(axis=-1)


def planck(frequency_hz: NDArray, temperature_k: NDArray) -> NDArray:
    """Bν(T) в Вт/(м²·ср·Гц), с пределом B₀ = 0."""
    exponent = PLANCK_CONSTANT * frequency_hz / (BOLTZMANN_CONSTANT * temperature_k)
    # Умножаем числитель и знаменатель на exp(-hν/kBT).
    # Так нет переполнения в экспоненте при большой частоте, а expm1
    # сохраняет точность разности 1 - exp(-hν/kBT) при малой частоте.
    prefactor = 2 * PLANCK_CONSTANT * frequency_hz**3 / LIGHT_SPEED**2
    return np.divide(
        prefactor * np.exp(-exponent),
        -np.expm1(-exponent),
        out=np.zeros_like(exponent),
        where=exponent > 0,
    )


@dataclass(frozen=True)
class TemperatureProfile:
    """T(ρ) = T0 + (Tw − T0)(ρ/R)^n. Класс, а не замыкание, чтобы передаваться в процессы."""
    T0: float
    Tw: float
    R: float
    n: float

    def __call__(self, rho: NDArray | float) -> NDArray:
        return self.T0 + (self.Tw - self.T0) * (np.asarray(rho) / self.R) ** self.n


def prepare_case(
    table: OpacityTable, T0: float, Tw: float, R: float, n: float,
    frequency_scale_hz: float, radius_fractions: tuple[float, ...],
    radial_steps: int, frequency_samples: int,
) -> tuple[NDArray, NDArray, NDArray, TemperatureProfile]:
    """Возвращает радиусы, частотные отсчёты, радиальную τ и профиль T."""
    if not np.isfinite([T0, Tw, R, n, frequency_scale_hz]).all():
        raise ValueError("Параметры модели должны быть конечными.")
    if R <= 0 or n <= 0 or frequency_scale_hz <= 0 or min(T0, Tw) <= 0:
        raise ValueError("R, n, температуры и масштаб частоты должны быть положительными.")
    if not table.covers(min(T0, Tw), max(T0, Tw)):
        raise ValueError(
            "Температурный профиль выходит за диапазон таблицы k "
            f"({table.temperatures_k[0]:g}–{table.temperatures_k[-1]:g} К). "
            "Задайте extrapolation='clamp' или 'log-linear' при загрузке таблицы."
        )
    if radial_steps < 1 or frequency_samples < 2:
        raise ValueError("Размеры сеток должны быть положительными, частотных отсчётов ≥ 2.")

    fractions = np.asarray(radius_fractions, dtype=float)
    if fractions.ndim != 1 or not fractions.size or not np.all((0 <= fractions) & (fractions <= 1)):
        raise ValueError("Точки наблюдения задаются непустым списком r/R от 0 до 1.")
    radii = R * fractions
    temperature = TemperatureProfile(T0, Tw, R, n)

    # τ_rad — отдельный интеграл вдоль радиуса, общий для обоих методов.
    # Формула средних прямоугольников берёт k в середине каждого Δr.
    dr = R / radial_steps
    tau_temperature = temperature((np.arange(radial_steps) + 0.5) * dr)
    tau_radial = dr * table.interpolate_all(tau_temperature).sum(axis=0)

    # На стыке соседних полос могут быть два разных односторонних значения
    # спектра. Храним частотные отсчёты каждой полосы отдельно.
    nu_star = np.linspace(table.bands[:, 0], table.bands[:, 1], frequency_samples, axis=1)
    return radii, nu_star, tau_radial, temperature
