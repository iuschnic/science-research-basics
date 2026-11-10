"""Графики спектров обоих методов и их расхождения."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from transfer_common import TransferResult

RELATIVE_ERROR_FLOOR = 1e-8


def _save_spectra(
    path: Path, nu_star: NDArray, curves: list[tuple[str, NDArray]],
    x_label: str, title: str, *, error_plot: bool = False,
) -> None:
    left, right = nu_star[:, 0], nu_star[:, -1]
    shared = right[:-1] == left[1:]
    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    has_jumps = False
    for index, (label, spectrum) in enumerate(curves):
        color = f"C{index % 10}"
        lines = ax.plot(nu_star.T, spectrum.T, color=color, linewidth=1.9)
        lines[0].set_label(label)
        # Полосы имеют независимые односторонние значения. Соединяем
        # пунктиром только соседние конечные значения на общей частоте.
        before, after = spectrum[:-1, -1], spectrum[1:, 0]
        valid = shared & np.isfinite(before) & np.isfinite(after)
        if valid.any():
            has_jumps = True
            ax.vlines(left[1:][valid], before[valid], after[valid],
                      color=color, linestyles="dotted", linewidth=1.25)
    if has_jumps:
        ax.plot([], [], color="0.35", linestyle=":", label="Скачок на границе полосы")
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set(xlabel=x_label, title=title, xlim=(left[0], right[-1]))
    if error_plot:
        ax.set_ylabel("Относительная ошибка, %")
        # Линейная шкала до 1%, логарифмическая выше для сравнения пиков.
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.text(
            0.02, 0.98, r"Пропуски при $|q_{\rm точн}|\leq 10^{-8}\max|q_{\rm точн}|$",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
        )
        if not any(np.isfinite(spectrum).any() for _, spectrum in curves):
            ax.text(0.5, 0.5, "Ошибка в процентах не определена",
                    transform=ax.transAxes, ha="center")
    else:
        ax.set_ylabel(r"$\mathrm{div}\,F_\nu$ (Вт/(м³·Гц))")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.25)
    ax.legend(loc="best" if not error_plot else "lower right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_tau(path: Path, result: TransferResult, x_label: str) -> None:
    figure, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    left, right = result.nu_star[:, 0], result.nu_star[:, -1]
    tau = result.tau_radial
    ax.hlines(tau, left, right, color="C0", linewidth=2)
    shared = right[:-1] == left[1:]
    ax.vlines(left[1:][shared], tau[:-1][shared], tau[1:][shared], color="C0")
    if (tau > 0).all():
        ax.set_yscale("log")
    ax.axhline(1, color="0.35", linestyle="--", linewidth=0.8, label=r"$\tau_\nu=1$")
    ax.set(
        xlabel=x_label,
        ylabel=r"$\tau_\nu$ (безразмерная)",
        title="Оптическая толщина от оси до поверхности",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _relative_error_percent(reference: NDArray, approx: NDArray) -> NDArray:
    """Абсолютное относительное расхождение на одном радиусе."""
    magnitude = np.abs(reference)
    valid = magnitude > RELATIVE_ERROR_FLOOR * magnitude.max()
    return np.divide(
        100 * np.abs(approx - reference), magnitude,
        out=np.full_like(reference, np.nan), where=valid,
    )


def save_comparison_plots(
    exact: TransferResult, approx: TransferResult, output_dir: Path,
    radius_m: float, frequency_scale_hz: float,
) -> None:
    if not (np.array_equal(exact.nu_star, approx.nu_star)
            and np.array_equal(exact.radii_m, approx.radii_m)
            and np.array_equal(exact.tau_radial, approx.tau_radial)):
        raise ValueError("Для сравнения требуются общие радиусы, частоты и оптическая толщина.")
    mantissa, _, exponent = f"{frequency_scale_hz:g}".partition("e")
    scale_label = mantissa
    if exponent:
        scale_label = rf"10^{{{int(exponent)}}}"
        if mantissa != "1":
            scale_label = rf"{mantissa}\times {scale_label}"
    x_label = rf"$\nu^* = \nu / ({scale_label}\,\mathrm{{Hz}})$"
    nu_star = exact.nu_star
    fractions = exact.radii_m / radius_m
    radius_labels = [rf"$r/R = {fraction:g}$" for fraction in fractions]
    filenames = [
        f"div_flux_{i + 1:02d}_r_over_R_{fraction:g}.png"
        for i, fraction in enumerate(fractions)
    ]

    methods = (
        ("exact", "Точный метод", exact),
        ("schuster_schwarzschild", "Шустер — Шварцшильд", approx),
    )
    for folder, label, result in methods:
        method_dir = output_dir / folder
        by_radius = method_dir / "by_radius"
        by_radius.mkdir(parents=True, exist_ok=True)
        _save_spectra(
            method_dir / "div_flux.png", nu_star,
            list(zip(radius_labels, result.div_flux)), x_label,
            f"Спектр дивергенции потока · {label}",
        )
        _save_tau(method_dir / "tau_radial.png", result, x_label)
        for name, radius_label, spectrum in zip(filenames, radius_labels, result.div_flux):
            _save_spectra(
                by_radius / name, nu_star, [(radius_label, spectrum)], x_label,
                f"Дивергенция при {radius_label} · {label}",
            )

    values_dir = output_dir / "comparison" / "values"
    errors_dir = output_dir / "comparison" / "error_percent"
    values_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    for name, label, reference, approximation in zip(
        filenames, radius_labels, exact.div_flux, approx.div_flux
    ):
        _save_spectra(
            values_dir / name, nu_star,
            [("Точный метод", reference), ("Шустер — Шварцшильд", approximation)],
            x_label, f"Сравнение дивергенции при {label}",
        )
        _save_spectra(
            errors_dir / name, nu_star,
            [("ШШ относительно точного", _relative_error_percent(reference, approximation))],
            x_label, f"Ошибка приближения при {label}", error_plot=True,
        )
