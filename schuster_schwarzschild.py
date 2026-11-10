"""Цилиндрическое полусферическое приближение Шустера — Шварцшильда."""

import numpy as np
from numpy.typing import NDArray

from opacity_table import OpacityTable
from transfer_common import TransferResult, planck, prepare_case


def _solve_radial(
    faces: NDArray, optical_steps: NDArray, source: NDArray,
) -> tuple[NDArray, NDArray]:
    """Решает двухпоточные уравнения на ячейках с постоянными k и B."""
    # x = r/R, S = πB, a = 2 R k. Излучение при минимальном радиусе
    # переходит из внутреннего полупотока в наружный. Берём касательное
    # значение со стороны, откуда оно приходит: I_t ≈ I⁻ и f ≈ F⁻/r.
    # Тогда dF⁻/dx = a (F⁻ − S), а d(xF⁺)/dx = a x (S − F⁺) + F⁻.
    tau = optical_steps[..., None]
    transmission = np.exp(-tau)
    emitted = -np.expm1(-tau)
    inward = np.zeros((len(faces), *source.shape[1:]))
    outward = np.empty_like(inward)

    # Входящий полупоток известен на открытой поверхности. Движемся к оси
    # в направлении распространения этой группы, сохраняя F⁻ ≥ 0.
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
        # Сгруппированные слагаемые неотрицательны, в том числе при k=0.
        weighted_outward = (
            left * transmission[i] * outward[i]
            + (left * emitted[i] + width * one_minus_average[i]) * source[i]
            + width * average[i] * inward[i + 1]
        )
        outward[i + 1] = weighted_outward / right
    return outward, inward


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
) -> TransferResult:
    """Решает двухпоточные уравнения на радиальных ячейках."""
    if flow_cells < 1:
        raise ValueError("Число радиальных ячеек ШШ должно быть положительным.")
    radii, nu_star, tau_radial, temperature = prepare_case(
        table, T0, Tw, R, n, frequency_scale_hz, radius_fractions,
        radial_steps, frequency_samples,
    )
    frequency_hz = nu_star * frequency_scale_hz
    # Включаем точки наблюдения в границы ячеек, чтобы сохранять
    # рассчитанные там полупотоки без дополнительной интерполяции.
    x = radii / R
    faces = np.unique(np.concatenate((np.linspace(0, 1, flow_cells + 1), x)))
    centers = (faces[:-1] + faces[1:]) / 2
    cell_temperature = temperature(R * centers)
    cell_k = np.column_stack([
        table.interpolate(cell_temperature, band) for band in range(len(table.bands))
    ])
    source = np.pi * planck(frequency_hz, cell_temperature[:, None, None])
    tau_steps = (2 * R * np.diff(faces))[:, None] * cell_k
    outward, inward = _solve_radial(faces, tau_steps, source)
    indices = np.searchsorted(faces, x)
    outward, inward = outward[indices], inward[indices]

    local_temperature = temperature(radii)
    local_k = np.column_stack([
        table.interpolate(local_temperature, band) for band in range(len(table.bands))
    ])
    local_source = np.pi * planck(frequency_hz, local_temperature[:, None, None])
    # Вычитание потоковых уравнений устраняет f. Получаем локальный
    # баланс div F = 2k(2πB − F⁺ − F⁻), применимый и на оси.
    div_flux = 2 * local_k[..., None] * (2 * local_source - outward - inward)
    return TransferResult(nu_star, tau_radial, radii, outward, inward, div_flux)
