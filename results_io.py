"""Выгрузка численных результатов в текстовые таблицы (для отчёта, Excel, Origin)."""

from pathlib import Path

import numpy as np

from transfer_common import TransferResult, integrate_over_frequency


def _write(path: Path, header: str, columns: list[np.ndarray], fmt: str = "%.6e") -> None:
    np.savetxt(path, np.column_stack(columns), fmt=fmt, delimiter="\t",
               header=header, comments="# ", encoding="utf-8")


def save_results_txt(
    exact: TransferResult, approx: TransferResult, output_dir: Path,
    radius_m: float, frequency_scale_hz: float, description: str,
) -> None:
    """Пишет в output_dir/txt:

    * tau_radial.txt — оптическая толщина каждой полосы;
    * div_flux_exact.txt, div_flux_ss.txt — спектры div F_ν (строка = частотный
      отсчёт, столбцы = точки r/R), Вт/(м³·Гц);
    * flux_exact.txt, flux_ss.txt — результирующий поток F_ν = F⁺ − F⁻, Вт/(м²·Гц);
    * integrated.txt — Q(r) = ∫div F_ν dν обоих методов, их отношение и разность.
    """
    folder = output_dir / "txt"
    folder.mkdir(parents=True, exist_ok=True)
    head = f"{description}\n"
    fractions = exact.radii_m / radius_m
    bands = len(exact.tau_radial)
    samples = exact.nu_star.shape[1]

    left, right = exact.nu_star[:, 0], exact.nu_star[:, -1]
    _write(folder / "tau_radial.txt",
           head + "полоса\tnu*_лев\tnu*_прав\ttau_rad",
           [np.arange(bands), left, right, exact.tau_radial])

    band_index = np.repeat(np.arange(bands), samples)
    nu_flat = exact.nu_star.ravel()
    radius_header = "\t".join(f"r/R={x:g}" for x in fractions)
    for name, result in (("exact", exact), ("ss", approx)):
        _write(folder / f"div_flux_{name}.txt",
               head + f"div F_nu, Вт/(м3·Гц)\nполоса\tnu*\t{radius_header}",
               [band_index, nu_flat, *result.div_flux.reshape(len(fractions), -1)])
        _write(folder / f"flux_{name}.txt",
               head + f"F_nu = F+ - F-, Вт/(м2·Гц)\nполоса\tnu*\t{radius_header}",
               [band_index, nu_flat, *result.flux.reshape(len(fractions), -1)])

    q_exact = integrate_over_frequency(exact.div_flux, exact.nu_star, frequency_scale_hz)
    q_approx = integrate_over_frequency(approx.div_flux, approx.nu_star, frequency_scale_hz)
    ratio = np.divide(q_approx, q_exact, out=np.full_like(q_exact, np.nan), where=q_exact != 0)
    norm_diff = 100 * (q_approx - q_exact) / np.abs(q_exact).max()
    _write(folder / "integrated.txt",
           head + "Q = ∫div F_nu dnu, Вт/м3\n"
                  "r/R\tQ_точн\tQ_ШШ\tQ_ШШ/Q_точн\t(Q_ШШ-Q_точн)/max|Q_точн|, %",
           [fractions, q_exact, q_approx, ratio, norm_diff])
