"""Перенос по прямым лучам в бесконечном цилиндре без рассеяния и подсветки."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from opacity_table import OpacityTable
from transfer_common import TemperatureProfile, TransferResult, planck as _planck, prepare_case

THICK_BAND_TAU = 100.0
THICK_BAND_REFINEMENT = 4


def _directions(polar_order: int, azimuth_order: int) -> tuple[NDArray, ...]:
    """Узлы направлений и веса интегрирования по полному телесному углу."""
    polar_nodes, polar_weights = leggauss(polar_order)
    azimuth_nodes, azimuth_weights = leggauss(azimuth_order)

    # Заменяем θ на μz = cos θ, тогда dΩ = dμz dφ с положительными
    # пределами интегрирования. Отражения по z и по y оставляют T вдоль
    # луча прежней. Поэтому считаем лишь μz ∈ [0, 1] и φ ∈ [0, π],
    # а вес каждого направления увеличиваем в четыре раза.
    mu_z = (polar_nodes + 1) / 2
    # Делим азимут в π/2, где меняется знак радиальной проекции.
    # Так F+ и F− интегрируются на своих участках без узлов на границе.
    phi_first_half = (azimuth_nodes + 1) * np.pi / 4
    phi = np.concatenate((phi_first_half, np.pi - phi_first_half))
    phi_weights = np.tile(azimuth_weights * np.pi / 4, 2)
    sin_theta, phi_grid = np.broadcast_arrays(
        np.sqrt(1 - mu_z[:, None] ** 2), phi[None, :]
    )
    weights = 4 * (polar_weights[:, None] / 2) * phi_weights[None, :]
    return sin_theta.ravel(), np.cos(phi_grid).ravel(), np.sin(phi_grid).ravel(), weights.ravel()


def _ray_midpoints(
    r: float, R: float, sin_theta: NDArray, cos_phi: NDArray,
    sin_phi: NDArray, steps: int,
) -> tuple[NDArray, NDArray]:
    """Радиусы середин и длины отрезков с осями [отрезок, луч]."""
    # P = (r, 0, 0). Смотрим назад от P против движения света.
    # При поперечном смещении x точка луча имеет координаты
    # (r - x cos φ, -x sin φ). Пересечение с окружностью радиуса R
    # даёт d = r cos φ + sqrt(R² - r² sin²φ), длину проекции пути.
    radial_projection = r * cos_phi
    remaining = (R - r) * (R + r)
    root = np.sqrt(remaining + radial_projection**2)
    distance_xy = root + radial_projection
    # При взгляде назад на ближнюю сторону вычитание близких чисел
    # теряет точность. Домножаем выражение на сопряжённое.
    near_side = cos_phi < 0
    distance_xy[near_side] = remaining / (root - radial_projection)[near_side]

    # В координате x свет идёт от границы x=d к точке P при x=0.
    # Сгущаем границы отрезков к P кубическим преобразованием. В очень
    # плотной полосе до P доходит свет с пути порядка 1/k, и равномерная
    # сетка может пропустить изменение температуры на этом масштабе.
    fractions = np.linspace(1.0, 0.0, steps + 1) ** 3
    midpoints = (fractions[:-1] + fractions[1:]) / 2
    x = midpoints[:, None] * distance_xy[None, :]
    midpoint_radii = np.hypot(r - x * cos_phi, x * sin_phi)
    # Наклон к оси растягивает путь. Узлы Гаусса лежат внутри интервала
    # μz, поэтому строго осевые лучи с sin θ = 0 в сумму не попадают.
    ds = -np.diff(fractions)[:, None] * (distance_xy / sin_theta)[None, :]
    return np.minimum(midpoint_radii, R), ds


def _trace_band(
    frequency_hz: NDArray, ray_temperature: NDArray, ray_k: NDArray,
    ds: NDArray, local_source: NDArray,
) -> tuple[NDArray, NDArray]:
    """Интенсивность I и разность Bν(T(P)) − I на конце каждого луча."""
    # В ЛТР jν = kν Bν. Для сегмента с постоянными k и B точное
    # решение dI/ds = k(B-I) даёт Iout = Iin exp(-Δτ) + B(1-exp(-Δτ)).
    # Источник B зависит от частоты даже там, где k один на всю полосу.
    segment_source = _planck(frequency_hz, ray_temperature[..., None])
    intensity = np.zeros_like(segment_source[0])  # На входе Iν = 0.
    deficit = np.broadcast_to(local_source, intensity.shape).copy()
    optical_steps = ray_k * ds
    for source, optical_step in zip(segment_source, optical_steps):
        delta_tau = optical_step[:, None]
        transmission = np.exp(-delta_tau)
        emission_weight = -np.expm1(-delta_tau)
        intensity = intensity * transmission + source * emission_weight

        # Та же формула для D = Bν(T(P)) − I. Считаем D отдельно, чтобы
        # в непрозрачной изотермической области не вычитать два почти
        # одинаковых числа при вычислении малой дивергенции.
        deficit = deficit * transmission + (local_source - source) * emission_weight
    return intensity, deficit


def solve_exact(
    table: OpacityTable,
    T0: float,
    Tw: float,
    R: float,
    n: float,
    *,
    frequency_scale_hz: float = 1e15,
    radius_fractions: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0),
    polar_order: int = 20,
    azimuth_order: int = 24,
    ray_steps: int = 192,
    radial_steps: int = 4096,
    frequency_samples: int = 9,
    workers: int | None = 1,
) -> TransferResult:
    """Вычисляет спектр в заданных точках. T0, Tw в К, R в м, ν = ν* × масштаб.

    workers — число процессов (точки наблюдения независимы). None — по числу
    ядер. При workers > 1 вызов должен стоять под `if __name__ == "__main__":`.
    """
    if min(polar_order, azimuth_order, ray_steps) < 1:
        raise ValueError("Размеры угловых сеток и сетки луча должны быть положительными.")
    radii, nu_star, tau_radial, temperature = prepare_case(
        table, T0, Tw, R, n, frequency_scale_hz, radius_fractions,
        radial_steps, frequency_samples,
    )
    frequency_hz = nu_star * frequency_scale_hz
    directions = _directions(polar_order, azimuth_order)
    tasks = [
        (table, temperature, r, R, frequency_hz, tau_radial, directions, ray_steps)
        for r in radii
    ]
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        parts = [_solve_point(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_solve_point, *zip(*tasks)))
    flux_plus, flux_minus, div_flux = (np.stack(values) for values in zip(*parts))
    return TransferResult(nu_star, tau_radial, radii, flux_plus, flux_minus, div_flux)


def _solve_point(
    table: OpacityTable, temperature: TemperatureProfile, r: float, R: float,
    frequency_hz: NDArray, tau_radial: NDArray, directions: tuple[NDArray, ...], ray_steps: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """F⁺, F⁻ и div F во всех полосах для одной точки наблюдения r."""
    sin_theta, cos_phi, sin_phi, angular_weights = directions
    mu_r = sin_theta * cos_phi
    outward_weights = (angular_weights * np.maximum(mu_r, 0))[:, None]
    inward_weights = (angular_weights * np.maximum(-mu_r, 0))[:, None]
    shape = frequency_hz.shape
    flux_plus, flux_minus, div_flux = np.empty(shape), np.empty(shape), np.empty(shape)

    ray_grids = {}
    local_temperature = temperature(r)
    local_k = table.interpolate_all(np.atleast_1d(local_temperature))[0]
    for band, frequencies in enumerate(frequency_hz):
        # В очень плотных полосах дополнительно уточняем отрезки.
        # Геометрию переиспользуем во всех полосах с той же сеткой.
        steps = ray_steps
        if tau_radial[band] >= THICK_BAND_TAU:
            steps *= THICK_BAND_REFINEMENT
        if steps not in ray_grids:
            midpoint_radii, ds = _ray_midpoints(r, R, sin_theta, cos_phi, sin_phi, steps)
            ray_grids[steps] = temperature(midpoint_radii), ds
        ray_temperature, ds = ray_grids[steps]
        local_source = _planck(frequencies, local_temperature)
        intensity, deficit = _trace_band(
            frequencies, ray_temperature, table.interpolate(ray_temperature, band),
            ds, local_source,
        )
        # Поток учитывает проекцию на радиус. Оба полупотока записаны
        # положительными. Чистый радиальный поток равен F+ − F−.
        flux_plus[band] = (intensity * outward_weights).sum(axis=0)
        flux_minus[band] = (intensity * inward_weights).sum(axis=0)

        # Интегрируем уравнение переноса по всей сфере направлений.
        # div Fν = (1/r) d[r(F+ − F−)]/dr = 4πjν − kν∫Iν dΩ.
        # При jν=kνBν считаем kν∫(Bν−Iν)dΩ. Эта форма не требует
        # численного дифференцирования потока и применима на оси.
        div_flux[band] = local_k[band] * (deficit * angular_weights[:, None]).sum(axis=0)
    return flux_plus, flux_minus, div_flux
