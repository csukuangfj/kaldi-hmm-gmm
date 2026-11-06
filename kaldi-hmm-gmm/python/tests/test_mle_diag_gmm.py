#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_mle_diag_gmm_py
import unittest

import kaldi_hmm_gmm as khg
import numpy as np


class TestAccumDiagGmm(unittest.TestCase):
    def test_MleDiagGmmOptions_default_constructor(self):
        opts = khg.MleDiagGmmOptions()
        assert abs(opts.min_gaussian_weight - 1e-5) < 1e-5
        assert abs(opts.min_gaussian_occupancy - 10) < 1e-5
        assert abs(opts.min_variance - 0.001) < 1e-5
        assert opts.remove_low_count_gaussians is True
        print(opts)

    def test_MleDiagGmmOptions_constructor(self):
        opts = khg.MleDiagGmmOptions(
            min_gaussian_weight=1,
            min_gaussian_occupancy=2,
            min_variance=3,
            remove_low_count_gaussians=False,
        )
        assert abs(opts.min_gaussian_weight - 1) < 1e-5
        assert abs(opts.min_gaussian_occupancy - 2) < 1e-5
        assert abs(opts.min_variance - 3) < 1e-5
        assert opts.remove_low_count_gaussians is False
        print(opts)

    def test_MapDiagGmmOptions_default_constructor(self):
        opts = khg.MapDiagGmmOptions()
        assert abs(opts.mean_tau - 10) < 1e-5
        assert abs(opts.variance_tau - 50) < 1e-5
        assert abs(opts.weight_tau - 10) < 1e-5
        print(opts)

    def test_MapDiagGmmOptions_non_default_constructor(self):
        opts = khg.MapDiagGmmOptions(mean_tau=1, variance_tau=2, weight_tau=3)
        assert abs(opts.mean_tau - 1) < 1e-5
        assert abs(opts.variance_tau - 2) < 1e-5
        assert abs(opts.weight_tau - 3) < 1e-5
        print(opts)

    def test(self):
        num_gauss = 3
        dim = 5
        acc = khg.AccumDiagGmm()
        # All -> weight, mean, variance
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmAll)
        assert acc.flags == khg.GmmUpdateFlags.kGmmAll
        assert acc.num_gauss == num_gauss
        assert acc.dim == dim
        assert acc.occupancy.shape == (num_gauss,)
        assert acc.mean_accumulator.shape == (num_gauss, dim)
        assert acc.variance_accumulator.shape == (num_gauss, dim)

        assert acc.occupancy.dtype == np.float64, acc.occupancy.dtype
        assert acc.mean_accumulator.dtype == np.float64, acc.mean_accumulator.dtype
        assert (
            acc.variance_accumulator.dtype == np.float64
        ), acc.variance_accumulator.dtype

        # kGmmWeights -> Only update weights
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmWeights)
        assert acc.flags == khg.GmmUpdateFlags.kGmmWeights
        assert acc.num_gauss == num_gauss
        assert acc.dim == dim
        assert acc.occupancy.shape == (num_gauss,)
        assert len(acc.mean_accumulator) == 0, acc.mean_accumulator
        assert len(acc.variance_accumulator) == 0, acc.variance_accumulator

        # Even if we only specify Means, it will also update weights
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmMeans)
        assert acc.num_gauss == num_gauss
        assert acc.dim == dim
        assert acc.occupancy.shape == (num_gauss,)
        assert acc.mean_accumulator.shape == (num_gauss, dim)
        assert len(acc.variance_accumulator) == 0, acc.variance_accumulator

        # Even if we only specify Variances, it will also update means, and weights
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmVariances)
        assert acc.num_gauss == num_gauss
        assert acc.dim == dim
        assert acc.occupancy.shape == (num_gauss,)
        assert acc.mean_accumulator.shape == (num_gauss, dim)
        assert acc.variance_accumulator.shape == (num_gauss, dim)

    def test_accumulate_for_component(self):
        num_gauss = 3
        dim = 5
        acc = khg.AccumDiagGmm()
        # All -> weight, mean, variance
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmAll)
        d = np.random.rand(dim).astype(np.float32)
        comp_index = 1
        weight = 0.25
        acc.accumulate_for_component(data=d, comp_index=comp_index, weight=weight)

        assert acc.flags == khg.GmmUpdateFlags.kGmmAll
        assert np.allclose(
            np.asarray(acc.occupancy), np.asarray([0, weight, 0]).astype(np.float64)
        ), acc.occupancy
        assert np.allclose(
            np.asarray(acc.mean_accumulator[comp_index]), d.astype(np.float64) * weight
        )

        assert np.allclose(
            np.asarray(acc.variance_accumulator[comp_index]),
            (d ** 2).astype(np.float64) * weight,
        )

        d0 = np.random.rand(dim).astype(np.float32)
        comp_index0 = 0
        weight0 = 0.35
        acc.accumulate_for_component(data=d0, comp_index=comp_index0, weight=weight0)

        assert np.allclose(
            np.asarray(acc.occupancy), np.asarray([weight0, weight, 0]).astype(np.float64)
        ), acc.occupancy

        assert np.allclose(
            np.asarray(acc.mean_accumulator[comp_index0]), d0.astype(np.float64) * weight0
        )

        assert np.allclose(
            np.asarray(acc.variance_accumulator[comp_index0]),
            (d0 ** 2).astype(np.float64) * weight0,
        )

        f = 0.1
        acc.scale(f=0.1, flags=khg.GmmUpdateFlags.kGmmAll)

        assert np.allclose(
            np.asarray(acc.occupancy),
            np.asarray([weight0, weight, 0]).astype(np.float64) * f,
        ), acc.occupancy

        assert np.allclose(
            np.asarray(acc.mean_accumulator[comp_index]), d.astype(np.float64) * weight * f
        )
        assert np.allclose(
            np.asarray(acc.mean_accumulator[comp_index0]),
            d0.astype(np.float64) * weight0 * f,
        )

        assert np.allclose(
            np.asarray(acc.variance_accumulator[comp_index]),
            (d ** 2).astype(np.float64) * weight * f,
        )

        assert np.allclose(
            np.asarray(acc.variance_accumulator[comp_index0]),
            (d0 ** 2).astype(np.float64) * weight0 * f,
        )

        acc.set_zero(khg.GmmUpdateFlags.kGmmAll)
        assert acc.occupancy.sum() == 0
        assert np.abs(np.asarray(acc.mean_accumulator)).sum().item() == 0
        assert np.abs(np.asarray(acc.variance_accumulator)).sum().item() == 0

    def test_accumulate_from_posteriors(self):
        num_gauss = 3
        dim = 5
        acc = khg.AccumDiagGmm()
        # All -> weight, mean, variance
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmAll)

        d = np.random.rand(dim).astype(np.float32)
        comp_index = 1
        weight = 0.25
        acc.accumulate_for_component(data=d, comp_index=comp_index, weight=weight)

        occupancy = np.asarray(acc.occupancy).copy()
        mean_accumulator = np.asarray(acc.mean_accumulator).copy()
        variance_accumulator = np.asarray(acc.variance_accumulator).copy()

        data = np.random.rand(dim).astype(np.float32)
        posteriors = np.random.rand(num_gauss).astype(np.float32)
        acc.accumulate_from_posteriors(data=data, gauss_posteriors=posteriors)

        expected_occupancy = occupancy + posteriors
        assert np.allclose(np.asarray(acc.occupancy), expected_occupancy)

        expected_mean_accumulator = mean_accumulator + np.expand_dims(posteriors, 1) * data
        assert np.allclose(
            np.asarray(acc.mean_accumulator), expected_mean_accumulator
        )

        expected_variance_accumulator = (
            variance_accumulator + np.expand_dims(posteriors, 1) * (data ** 2)
        )
        assert np.allclose(
            np.asarray(acc.variance_accumulator), expected_variance_accumulator
        )

    def test_accumulate_from_diag(self):
        num_gauss = 3
        dim = 5
        acc = khg.AccumDiagGmm()
        # All -> weight, mean, variance
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmAll)

        d = np.random.rand(dim).astype(np.float32)
        comp_index = 1
        weight = 0.25
        acc.accumulate_for_component(data=d, comp_index=comp_index, weight=weight)

        occupancy = np.asarray(acc.occupancy).copy()
        mean_accumulator = np.asarray(acc.mean_accumulator).copy()
        variance_accumulator = np.asarray(acc.variance_accumulator).copy()

        data = np.random.rand(dim).astype(np.float32)

        diag_gmm = khg.DiagGmm(nmix=num_gauss, dim=dim)
        weights = np.random.rand(num_gauss).astype(np.float32)
        weights /= weights.sum()

        mean = np.random.rand(num_gauss, dim)
        var = np.random.rand(num_gauss, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        w = 0.2
        log_like = acc.accumulate_from_diag(gmm=diag_gmm, data=data, weight=w)

        expected_log_like, posteriors = diag_gmm.component_posteriors(data)
        posteriors = np.asarray(posteriors)
        assert abs(log_like - expected_log_like) < 1e-5

        # similar to accumulate_from_posteriors()

        posteriors *= w  # scale it!
        expected_occupancy = occupancy + posteriors
        assert np.allclose(np.asarray(acc.occupancy), expected_occupancy)

        expected_mean_accumulator = mean_accumulator + np.expand_dims(posteriors, 1) * data
        assert np.allclose(
            np.asarray(acc.mean_accumulator), expected_mean_accumulator
        )

        expected_variance_accumulator = (
            variance_accumulator + np.expand_dims(posteriors, 1) * (data ** 2)
        )
        assert np.allclose(
            np.asarray(acc.variance_accumulator), expected_variance_accumulator
        )

    def test_add_stats_for_component(self):
        num_gauss = 3
        dim = 5
        acc = khg.AccumDiagGmm()
        # All -> weight, mean, variance
        acc.resize(num_gauss=num_gauss, dim=dim, flags=khg.GmmUpdateFlags.kGmmAll)

        d = np.random.rand(dim).astype(np.float32)
        comp_index = 1
        weight = 0.25
        acc.accumulate_for_component(data=d, comp_index=comp_index, weight=weight)

        occupancy = np.asarray(acc.occupancy).copy()
        mean_accumulator = np.asarray(acc.mean_accumulator).copy()
        variance_accumulator = np.asarray(acc.variance_accumulator).copy()

        occ = 0.3
        g = 0
        x_stats = np.random.rand(dim).astype(np.float64)
        x2_stats = np.random.rand(dim).astype(np.float64)

        acc.add_stats_for_component(g=g, occ=occ, x_stats=x_stats, x2_stats=x2_stats)

        occupancy[g] += occ
        mean_accumulator[g] += x_stats
        variance_accumulator[g] += x2_stats

        assert np.allclose(np.asarray(acc.occupancy), occupancy)
        assert np.allclose(np.asarray(acc.mean_accumulator), mean_accumulator)
        assert np.allclose(
            np.asarray(acc.variance_accumulator), variance_accumulator
        )


if __name__ == "__main__":
    np.random.seed(20230615)
    unittest.main()
