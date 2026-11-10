"""Проверки обоих методов. Запуск из корня проекта: python -m unittest discover -s tests"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_method import solve_exact  # noqa: E402
from opacity_table import OpacityTable, load_opacity_table  # noqa: E402
from schuster_schwarzschild import solve_schuster_schwarzschild  # noqa: E402
from transfer_common import integrate_over_frequency, planck  # noqa: E402

R = 0.01
FAST_EXACT = dict(polar_order=12, azimuth_order=12, ray_steps=96)


def one_band(k: float, band=(0.3, 0.4)) -> OpacityTable:
    """Одна полоса с k, не зависящим от T."""
    return OpacityTable(np.array([band]), np.array([1000.0, 20000.0]), np.array([[k], [k]]))


def two_bands_varying() -> OpacityTable:
    """Две полосы, k растёт с T: тонкая и умеренно толстая."""
    return OpacityTable(
        np.array([[0.2, 0.3], [0.3, 0.45]]),
        np.array([2000.0, 12000.0]),
        np.array([[0.5, 50.0], [5.0, 500.0]]),
    )


class TableReading(unittest.TestCase):
    def test_teacher_file_matches_clean_file(self):
        teacher = load_opacity_table(ROOT / "real_opacity.txt")
        clean = load_opacity_table(ROOT / "real_opacity_clean.txt")
        np.testing.assert_array_equal(teacher.bands, clean.bands)
        np.testing.assert_array_equal(teacher.temperatures_k, clean.temperatures_k)
        np.testing.assert_array_equal(teacher.coefficients_m_inv, clean.coefficients_m_inv)
        self.assertEqual(teacher.coefficients_m_inv.shape, (12, 193))

    def test_unit_scale(self):
        a = load_opacity_table(ROOT / "demo_opacity.txt")
        b = load_opacity_table(ROOT / "demo_opacity.txt", unit_scale=100)
        np.testing.assert_allclose(b.coefficients_m_inv, 100 * a.coefficients_m_inv)

    def test_interpolation_modes(self):
        t = OpacityTable(np.array([[0.1, 0.2]]), np.array([1000.0, 2000.0]), np.array([[1.0], [100.0]]))
        self.assertAlmostEqual(t.interpolate(1500.0, 0), 50.5)
        log = OpacityTable(t.bands, t.temperatures_k, t.coefficients_m_inv, interpolation="log")
        self.assertAlmostEqual(float(log.interpolate(1500.0, 0)), 10.0)

    def test_extrapolation_modes(self):
        base = (np.array([[0.1, 0.2]]), np.array([1000.0, 2000.0]), np.array([[1.0], [100.0]]))
        self.assertFalse(OpacityTable(*base).covers(500, 2000))
        clamp = OpacityTable(*base, extrapolation="clamp")
        self.assertAlmostEqual(float(clamp.interpolate(500.0, 0)), 1.0)
        loglin = OpacityTable(*base, extrapolation="log-linear")
        self.assertAlmostEqual(float(loglin.interpolate(500.0, 0)), 0.1)
        with self.assertRaises(ValueError):
            solve_schuster_schwarzschild(OpacityTable(*base), 2000, 500, R, 2.0)


class UniformCylinder(unittest.TestCase):
    def test_ss_matches_analytic_fluxes(self):
        for k in (1.0, 100.0, 1e4):
            res = solve_schuster_schwarzschild(one_band(k), 8000, 8000, R, 2.0,
                                               radius_fractions=(0.0, 0.5, 1.0))
            S = np.pi * planck(res.nu_star[0] * 1e15, 8000.0)
            r = res.radii_m[:, None]
            Fm = S * (1 - np.exp(-2 * k * (R - r)))
            frac = np.where(r > 0, -np.expm1(-4 * k * r) / np.where(r > 0, 4 * k * r, 1), 1.0)
            Fp = S * (1 - np.exp(-2 * k * (R - r)) * frac)
            np.testing.assert_allclose(res.flux_minus[:, 0], Fm, rtol=1e-10, atol=1e-12 * S.max())
            np.testing.assert_allclose(res.flux_plus[:, 0], Fp, rtol=1e-10, atol=1e-12 * S.max())

    def test_ss_divergence_has_no_cancellation_in_thick_medium(self):
        # Аналитика ШШ: q(0) = 4kπB·exp(−2kR). При kR = 100 это ~1e-87 от 4πkB.
        k = 1e4
        res = solve_schuster_schwarzschild(one_band(k), 8000, 8000, R, 2.0, radius_fractions=(0.0,))
        S = np.pi * planck(res.nu_star[0] * 1e15, 8000.0)
        expected = 4 * k * S * np.exp(-2 * k * R)
        np.testing.assert_allclose(res.div_flux[0, 0], expected, rtol=1e-8)

    def test_exact_thin_limit(self):
        # В оптически тонкой среде q → 4πkB, и оба метода совпадают.
        k = 1e-2
        ex = solve_exact(one_band(k), 8000, 8000, R, 2.0, radius_fractions=(0.0, 0.5), **FAST_EXACT)
        ss = solve_schuster_schwarzschild(one_band(k), 8000, 8000, R, 2.0, radius_fractions=(0.0, 0.5))
        q_thin = np.broadcast_to(4 * np.pi * k * planck(ex.nu_star[0] * 1e15, 8000.0), (2, 9))
        np.testing.assert_allclose(ex.div_flux[:, 0], q_thin, rtol=1e-3)
        np.testing.assert_allclose(ss.div_flux[:, 0], q_thin, rtol=1e-3)


class BoundaryAndBalance(unittest.TestCase):
    def setUp(self):
        self.table = two_bands_varying()
        self.fractions = tuple(np.linspace(0, 1, 41))

    def check(self, res, tol):
        np.testing.assert_allclose(res.flux_plus[0], res.flux_minus[0], rtol=1e-6)   # ось
        np.testing.assert_allclose(res.flux_minus[-1], 0.0, atol=1e-12 * res.flux_plus.max())  # стенка
        # ∫ q·2πr dr = 2πR·F(R) для интеграла по частоте
        Q = integrate_over_frequency(res.div_flux, res.nu_star, 1e15)
        F = integrate_over_frequency(res.flux, res.nu_star, 1e15)
        r = res.radii_m
        self.assertAlmostEqual(np.trapezoid(Q * 2 * np.pi * r, r) / (2 * np.pi * R * F[-1]), 1.0, delta=tol)

    def test_ss(self):
        self.check(solve_schuster_schwarzschild(self.table, 11000, 2000, R, 2.0,
                                                radius_fractions=self.fractions), 5e-3)

    def test_exact(self):
        self.check(solve_exact(self.table, 11000, 2000, R, 2.0, radius_fractions=self.fractions[::4],
                               **FAST_EXACT), 2e-2)


class ThickLimit(unittest.TestCase):
    def test_diffusion_limit_ratios(self):
        # Диффузионный предел. Вне оси ШШ даёт F = −(π/k)B′, точно F = −(4π/3k)B′ → 3/4.
        # На оси условие F⁺ = F⁻ даёт q = −(π/k)B″ против −(8π/3k)B″ → 3/8.
        table = one_band(1e5)
        T = (11000, 6000, R, 2.0)
        fr = (0.0, 0.3, 0.6)
        ex = solve_exact(table, *T, radius_fractions=fr, polar_order=24, azimuth_order=24, ray_steps=256)
        ss = solve_schuster_schwarzschild(table, *T, radius_fractions=fr)
        ratio = ss.div_flux[:, 0, 4] / ex.div_flux[:, 0, 4]
        np.testing.assert_allclose(ratio, [0.375, 0.75, 0.75], rtol=0.03)


class Parallel(unittest.TestCase):
    def test_workers_give_same_result(self):
        table = two_bands_varying()
        a = solve_exact(table, 11000, 2000, R, 2.0, radius_fractions=(0.0, 0.7), **FAST_EXACT)
        b = solve_exact(table, 11000, 2000, R, 2.0, radius_fractions=(0.0, 0.7), workers=2, **FAST_EXACT)
        np.testing.assert_array_equal(a.div_flux, b.div_flux)


if __name__ == "__main__":
    unittest.main()
