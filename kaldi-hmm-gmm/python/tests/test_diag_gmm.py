#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_diag_gmm_py


import math
import unittest

import torch

import kaldi_hmm_gmm as khg


class TestDiagGmm(unittest.TestCase):
    def test_get_set_remove(self):
        nmix = 10
        dim = 8

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        assert torch.allclose(diag_gmm.weights, weights)
        assert torch.allclose(diag_gmm.means, mean)
        assert torch.allclose(diag_gmm.vars, var)

        assert diag_gmm.num_gauss == nmix, diag_gmm.num_gauss
        assert diag_gmm.dim == dim, diag_gmm.dim

        assert diag_gmm.valid_gconsts is False
        num_bad = 0
        assert diag_gmm.compute_gconsts() == num_bad
        assert diag_gmm.valid_gconsts is True

        expected_gconsts = weights.log() - 0.5 * (
            diag_gmm.dim * math.log(2 * math.pi)
            + var.log().sum(dim=1)
            + mean.square().div(var).sum(dim=1)
        )

        assert torch.allclose(diag_gmm.means_invvars, mean.div(var))
        assert torch.allclose(diag_gmm.inv_vars, 1 / var)

        assert torch.allclose(diag_gmm.gconsts, expected_gconsts)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights.div_(weights.sum())
        for i, w in enumerate(weights.tolist()):
            diag_gmm.set_component_weight(i, w)

        assert diag_gmm.valid_gconsts is False
        assert diag_gmm.compute_gconsts() == num_bad

        for i in range(nmix):
            assert diag_gmm.weights[i] == weights[i]
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i])

        mean = torch.rand(nmix, dim, dtype=torch.float32)
        for i in range(nmix):
            diag_gmm.set_component_mean(i, mean[i])

        for i in range(nmix):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i])

        var = torch.rand(nmix, dim, dtype=torch.float32)
        for i in range(nmix):
            diag_gmm.set_component_inv_var(i, 1 / var[i])

        for i in range(nmix):
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i])

        mean = torch.rand(nmix, dim, dtype=torch.float32)
        var = torch.rand(nmix, dim, dtype=torch.float32)
        diag_gmm.set_invvars_and_means(1 / var, mean)

        assert torch.allclose(diag_gmm.means, mean)
        assert torch.allclose(diag_gmm.vars, var)
        for i in range(nmix):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i])

        diag_gmm.remove_component(0, renorm_weights=True)
        assert diag_gmm.num_gauss == nmix - 1
        assert diag_gmm.dim == dim
        assert torch.allclose(diag_gmm.weights, weights[1:] / weights[1:].sum())

        for i in range(nmix - 1):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i + 1])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i + 1])

        diag_gmm.remove_component(1, renorm_weights=False)
        assert diag_gmm.num_gauss == nmix - 2
        assert diag_gmm.dim == dim
        new_weights = weights[1:] / weights[1:].sum()
        new_weights = torch.cat([new_weights[0:1], new_weights[2:]])
        assert torch.allclose(diag_gmm.weights, new_weights)

        assert torch.allclose(diag_gmm.get_component_mean(0), mean[1])
        assert torch.allclose(diag_gmm.get_component_variance(0), var[1])
        for i in range(1, nmix - 2):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i + 2])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i + 2])

        diag_gmm.remove_components([0, 1, 2], renorm_weights=True)
        assert diag_gmm.num_gauss == nmix - 5

        for i in range(0, nmix - 5):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i + 5])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i + 5])

    def test_split(self):
        nmix = 1
        dim = 8

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        perturb_factor = 0.01
        new2old = diag_gmm.split(target_components=2, perturb_factor=0.01)  # noop
        assert diag_gmm.num_gauss == 2, diag_gmm.num_gauss
        assert new2old == [0]  # the second component is split from the first component

        assert diag_gmm.weights[0] == diag_gmm.weights[1]
        assert diag_gmm.weights[0] == weights[0] / 2

        assert torch.allclose(
            diag_gmm.get_component_variance(0),
            diag_gmm.get_component_variance(1),
        )

        assert torch.allclose(diag_gmm.get_component_variance(0), var[0])

        assert torch.allclose(diag_gmm.means.sum(dim=0), mean * 2)

    def test_split_2(self):
        nmix = 2
        dim = 8

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.tensor([0.4, 0.6], dtype=torch.float32)

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        perturb_factor = 0.01
        new2old = diag_gmm.split(target_components=4, perturb_factor=0.01)  # noop
        # First, we split the second component and get weight [0.4, 0.3, 0.3]
        # and then we split the first component and get weight [0.2, 0.3, 0.3, 0.2]
        assert diag_gmm.num_gauss == 4, diag_gmm.num_gauss
        assert new2old == [1, 0]
        # The first appended component is from component 1
        # The second appended component is from component 0
        expected_weights = torch.tensor(
            [weights[0] / 2, weights[1] / 2, weights[1] / 2, weights[0] / 2]
        )
        assert torch.allclose(diag_gmm.weights, expected_weights)
        assert torch.allclose(diag_gmm.means[0] + diag_gmm.means[3], mean[0] * 2)
        assert torch.allclose(diag_gmm.means[1] + diag_gmm.means[2], mean[1] * 2)
        assert torch.allclose(diag_gmm.vars[0], var[0])
        assert torch.allclose(diag_gmm.vars[3], var[0])

        assert torch.allclose(diag_gmm.vars[1], var[1])
        assert torch.allclose(diag_gmm.vars[2], var[1])

    def test_merge(self):
        nmix = 7
        dim = 6

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        history = diag_gmm.merge(target_components=1)
        assert history == []
        assert diag_gmm.weights[0] == 1
        assert diag_gmm.num_gauss == 1

        expected_means = torch.matmul(weights.unsqueeze(0), mean)
        assert torch.allclose(diag_gmm.means, expected_means)

        second_order_stats = torch.matmul(weights.unsqueeze(0), (var + mean.square()))
        expected_vars = second_order_stats - expected_means.square()
        assert torch.allclose(diag_gmm.vars, expected_vars)

    def test_merge_case_2(self):
        nmix = 4
        dim = 2

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.tensor(
            [
                [2, 2],
                [-10, -10],
                [1, 1],
                [-100, 100],
            ],
            dtype=torch.float32,
        )
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        history = diag_gmm.merge(target_components=3)
        # Only component 2 and 0 have the largest merged logdet
        # since they are the closes one
        #
        # 2 comes first before 0 in history since we are building
        # a lower triangular matrix in C++
        assert history == [2, 0]
        assert diag_gmm.num_gauss == 3

        assert diag_gmm.weights[0] == weights[1]
        assert diag_gmm.weights[1] == weights[2] + weights[0]
        assert diag_gmm.weights[2] == weights[3]

        assert torch.allclose(diag_gmm.means[0], mean[1])
        assert torch.allclose(
            diag_gmm.means[1],
            (weights[2] * mean[2] + weights[0] * mean[0]) / (weights[2] + weights[0]),
        )
        assert torch.allclose(diag_gmm.means[2], mean[3])

        assert torch.allclose(diag_gmm.vars[0], var[1])
        second_order_stats = weights[2] * (var[2] + mean[2].square())
        second_order_stats += weights[0] * (var[0] + mean[0].square())
        second_order_stats /= weights[2] + weights[0]
        expected_vars = (
            second_order_stats
            - (
                (weights[2] * mean[2] + weights[0] * mean[0])
                / (weights[2] + weights[0])
            ).square()
        )
        assert torch.allclose(diag_gmm.vars[1], expected_vars)
        assert torch.allclose(diag_gmm.vars[2], var[3])


if __name__ == "__main__":
    torch.manual_seed(20230414)
    unittest.main()
