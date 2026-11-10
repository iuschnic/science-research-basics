"""Цилиндрическое полусферическое приближение Шустера — Шварцшильда."""

import numpy as np
from numpy.typing import NDArray

from opacity_table import OpacityTable
from transfer_common import TransferResult, planck, prepare_case


def _solve_radial(
    faces: NDArray, optical_steps: NDArray, source: NDArray, surface_inward: NDArray | float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Решает двухпоточные уравнения на ячейках с постоянными k и источником.

    Уравнения линейны, поэтому одна и та же схема годится и для самих
    полупотоков (source = πB, на поверхности F⁻ = 0), и для дефицитов
    D± = πB(r₀) − F± (source = πB(r₀) − πB, на поверхности D⁻ = πB(r₀)).
    """
    # x = r/R, S = πB, a = 2 R k. Излучение при минимальном радиусе
    # переходит из внутреннего полупотока в наружный. Берём касательное
    # значение со стороны, откуда оно приходит: I_t ≈ I⁻ и f ≈ F⁻/r.
    # Тогда dF⁻/dx = a (F⁻ − S), а d(xF⁺)/dx = a x (S − F⁺) + F⁻.
    tau = optical_steps[..., None]
    transmission = np.exp(-tau)
    emitted = -np.expm1(-tau)
    inward = np.empty((len(faces), *source.shape[1:]))
    outward = np.empty_like(inward)
    inward[-1] = surface_inward

    # Входящая группа известна на открытой поверхности. Движемся к оси
    # в направлении её распространения.
    for i in range(len(source) - 1, -1, -1):
        inward[i] = transmission[i] * inward[i + 1] + emitted[i] * source[i]

    # A = (1 − exp(−2τ))/(2τ) — среднее затухание с удвоенной оптической
    # толщиной. Оно возникает при интегрировании F⁻(x) вдоль ячейки:
    # один множитель описывает изменение F⁻ к меньшему радиусу,
    # второй — перенос его вклада в F⁺ к внешней грани.
    average = np.divide(-np.expm1(-2 * tau), 2 * tau,
                        out=np.ones_like(tau), where=tau > 0)
    small = tau < 1e-4
    one_minus_average = 1 - average
    t = tau[small]
    one_minus_average[small] = t * (1 - t * (2 - t) / 3)

    outward[0] = inward[0]  # На оси полупотоки равны.
    for i, (left, right) in enumerate(zip(faces[:-1], faces[1:])):
        width = right - left
        # Интегрируем уравнение для xF⁺ с интегрирующим множителем.
        weighted_outward = (
            left * transmission[i] * outward[i]
            + (left * emitted[i] + width * one_minus_average[i]) * source[i]
            + width * average[i] * inward[i + 1]
        )
        outward[i + 1] = weighted_outward / right
    return outward, inward


def _faces_around(x0: float, flow_cells: int, refine_cells: int, min_width: float) -> NDArray:
    """Равномерная сетка + геометрическое сгущение к точке наблюдения x0 (в долях R).

    В толстой полосе в точку r₀ приходит свет с расстояний ~1/k. Источник
    в ячейке берётся в её середине, поэтому ячейки около r₀ должны быть
    много меньше 1/k, иначе сдвиг на полъячейки даёт ошибку порядка
    самой дивергенции.
    """
    offsets = np.geomspace(min_width, 1.0, refine_cells)
    faces = np.concatenate((
        np.linspace(0.0, 1.0, flow_cells + 1), [x0], x0 - offsets, x0 + offsets,
    ))
    return np.unique(np.clip(faces, 0.0, 1.0))


def solve_schuster_schwarzschild(
    table: OpacityTable,
    T0: float,
    Tw: float,
    R: float,
    n: float,
    *,
    frequency_scale_hz: float = 1e15,
    radius_fractions: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0),
    radial_steps: int = 4096,
    frequency_samples: int = 9,
    flow_cells: int = 1024,
    refine_cells: int = 240,
    min_cell_fraction: float = 1e-12,
) -> TransferResult:
    """Решает двухпоточные уравнения отдельно для каждой точки наблюдения."""
    if flow_cells < 1 or refine_cells < 2 or not 0 < min_cell_fraction < 1:
        raise ValueError("Неверные параметры радиальной сетки ШШ.")
    radii, nu_star, tau_radial, temperature = prepare_case(
        table, T0, Tw, R, n, frequency_scale_hz, radius_fractions,
        radial_steps, frequency_samples,
    )
    frequency_hz = nu_star * frequency_scale_hz
    band_count = len(table.bands)
    shape = (len(radii), band_count, frequency_samples)
    flux_plus = np.empty(shape)
    flux_minus = np.empty(shape)
    div_flux = np.empty(shape)

    for index, r0 in enumerate(radii):
        x0 = r0 / R
        faces = _faces_around(x0, flow_cells, refine_cells, min_cell_fraction)
        centers = (faces[:-1] + faces[1:]) / 2
        cell_temperature = temperature(R * centers)
        cell_k = table.interpolate_all(cell_temperature)            # [ячейка, полоса]
        source = np.pi * planck(frequency_hz, cell_temperature[:, None, None])
        tau_steps = (2 * R * np.diff(faces))[:, None] * cell_k
        at = np.searchsorted(faces, x0)

        # Сами полупотоки: нужны для графиков потока и баланса энергии.
        outward, inward = _solve_radial(faces, tau_steps, source)
        flux_plus[index], flux_minus[index] = outward[at], inward[at]

        # Дивергенция: div F = 2k(2πB₀ − F⁺ − F⁻) = 2k(D⁺ + D⁻), D± = πB₀ − F±.
        # Решаем те же уравнения сразу для D±. В толстой полосе F± ≈ πB₀,
        # и прямое вычитание теряло бы все значащие цифры.
        T_local = temperature(r0)
        local_source = np.pi * planck(frequency_hz, T_local)            # [полоса, отсчёт]
        deficit_out, deficit_in = _solve_radial(
            faces, tau_steps, local_source[None] - source, surface_inward=local_source,
        )
        local_k = table.interpolate_all(np.atleast_1d(T_local))[0]
        div_flux[index] = 2 * local_k[:, None] * (deficit_out[at] + deficit_in[at])

    return TransferResult(nu_star, tau_radial, radii, flux_plus, flux_minus, div_flux)
