#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_diag_gmm_py


import math
import pickle
import unittest

import kaldi_hmm_gmm as khg
import torch


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
        print(dir(diag_gmm))
        assert torch.allclose(torch.from_numpy(diag_gmm.weights), weights)

        diag_gmm.set_means(mean)
        assert torch.allclose(torch.from_numpy(diag_gmm.means), mean)

        diag_gmm.set_invvars(1 / var)
        assert torch.allclose(torch.from_numpy(diag_gmm.vars), var)

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

        assert torch.allclose(torch.from_numpy(diag_gmm.gconsts), expected_gconsts)

        assert torch.allclose(torch.from_numpy(diag_gmm.means_invvars), mean.div(var))

        assert torch.allclose(torch.from_numpy(diag_gmm.inv_vars), 1 / var)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights.div_(weights.sum())
        for i, w in enumerate(weights.tolist()):
            diag_gmm.set_component_weight(i, w)

        assert diag_gmm.valid_gconsts is False
        assert diag_gmm.compute_gconsts() == num_bad

        for i in range(nmix):
            assert diag_gmm.weights[i] == weights[i]
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i]
            )

            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i]
            )

        mean = torch.rand(nmix, dim, dtype=torch.float32)
        for i in range(nmix):
            diag_gmm.set_component_mean(i, mean[i])

        for i in range(nmix):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i]
            ), (i, diag_gmm.get_component_mean(i), mean[i])

        var = torch.rand(nmix, dim, dtype=torch.float32)
        for i in range(nmix):
            diag_gmm.set_component_inv_var(i, 1 / var[i])

        for i in range(nmix):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i]
            )

        mean = torch.rand(nmix, dim, dtype=torch.float32)
        var = torch.rand(nmix, dim, dtype=torch.float32)
        diag_gmm.set_invvars_and_means(1 / var, mean)

        assert torch.allclose(torch.from_numpy(diag_gmm.means), mean)
        assert torch.allclose(torch.from_numpy(diag_gmm.vars), var)

        for i in range(nmix):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i]
            )
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i]
            )

        diag_gmm.remove_component(0, renorm_weights=True)
        assert diag_gmm.num_gauss == nmix - 1
        assert diag_gmm.dim == dim
        assert torch.allclose(
            torch.from_numpy(diag_gmm.weights), weights[1:] / weights[1:].sum()
        )

        for i in range(nmix - 1):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i + 1]
            )

            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i + 1]
            )

        diag_gmm.remove_component(1, renorm_weights=False)
        assert diag_gmm.num_gauss == nmix - 2
        assert diag_gmm.dim == dim
        new_weights = weights[1:] / weights[1:].sum()
        new_weights = torch.cat([new_weights[0:1], new_weights[2:]])
        assert torch.allclose(torch.from_numpy(diag_gmm.weights), new_weights)

        assert torch.allclose(torch.from_numpy(diag_gmm.get_component_mean(0)), mean[1])
        assert torch.allclose(
            torch.from_numpy(diag_gmm.get_component_variance(0)), var[1]
        )

        for i in range(1, nmix - 2):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i + 2]
            )
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i + 2]
            )

        diag_gmm.remove_components([2, 1, 0], renorm_weights=True)
        assert diag_gmm.num_gauss == nmix - 5

        for i in range(0, nmix - 5):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i + 5]
            )
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i + 5]
            )

        assert diag_gmm.num_gauss == nmix - 5
        # remove the last commponent
        diag_gmm.remove_component(nmix - 5 - 1, renorm_weights=True)

        assert diag_gmm.num_gauss == nmix - 6

        for i in range(0, nmix - 6):
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_mean(i)), mean[i + 5]
            )
            assert torch.allclose(
                torch.from_numpy(diag_gmm.get_component_variance(i)), var[i + 5]
            )

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
            torch.from_numpy(diag_gmm.get_component_variance(0)),
            torch.from_numpy(diag_gmm.get_component_variance(1)),
        )

        assert torch.allclose(
            torch.from_numpy(diag_gmm.get_component_variance(0)), var[0]
        )

        assert torch.allclose(torch.from_numpy(diag_gmm.means).sum(dim=0), mean * 2)

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
        assert new2old == [1, 0], new2old
        # The first appended component is from component 1
        # The second appended component is from component 0
        expected_weights = torch.tensor(
            [weights[0] / 2, weights[1] / 2, weights[1] / 2, weights[0] / 2]
        )
        assert torch.allclose(torch.from_numpy(diag_gmm.weights), expected_weights)
        assert torch.allclose(
            torch.from_numpy(diag_gmm.means[0] + diag_gmm.means[3]), mean[0] * 2
        )
        assert torch.allclose(
            torch.from_numpy(diag_gmm.means[1] + diag_gmm.means[2]), mean[1] * 2
        )
        assert torch.allclose(torch.from_numpy(diag_gmm.vars[0]), var[0])
        assert torch.allclose(torch.from_numpy(diag_gmm.vars[3]), var[0])

        assert torch.allclose(torch.from_numpy(diag_gmm.vars[1]), var[1])
        assert torch.allclose(torch.from_numpy(diag_gmm.vars[2]), var[1])

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
        assert abs(diag_gmm.weights[0] - 1) < 1e-6, diag_gmm.weights
        assert diag_gmm.num_gauss == 1

        expected_means = torch.matmul(weights.unsqueeze(0), mean)
        assert torch.allclose(torch.from_numpy(diag_gmm.means), expected_means)

        second_order_stats = torch.matmul(weights.unsqueeze(0), (var + mean.square()))
        expected_vars = second_order_stats - expected_means.square()
        assert torch.allclose(torch.from_numpy(diag_gmm.vars), expected_vars)

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
        diag_gmm.compute_gconsts()

        if True:
            history = diag_gmm.merge(target_components=3)
            # Only component 2 and 0 have the largest merged logdet
            # since they are the closest one
            #
            # 2 comes first before 0 in history since we are building
            # a lower triangular matrix in C++
            assert history == [2, 0]
        else:
            diag_gmm.merge_kmeans(target_components=3)

        assert diag_gmm.num_gauss == 3

        assert diag_gmm.weights[0] == weights[1], diag_gmm.weights
        assert diag_gmm.weights[1] == weights[2] + weights[0]
        assert diag_gmm.weights[2] == weights[3]

        assert torch.allclose(torch.from_numpy(diag_gmm.means[0]), mean[1])
        assert torch.allclose(
            torch.from_numpy(diag_gmm.means[1]),
            (weights[2] * mean[2] + weights[0] * mean[0]) / (weights[2] + weights[0]),
        )
        assert torch.allclose(torch.from_numpy(diag_gmm.means[2]), mean[3])

        assert torch.allclose(torch.from_numpy(diag_gmm.vars[0]), var[1])

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
        assert torch.allclose(torch.from_numpy(diag_gmm.vars[1]), expected_vars)
        assert torch.allclose(torch.from_numpy(diag_gmm.vars[2]), var[3])

    def test_log_likes(self):
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
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        log_likes: float = diag_gmm.log_likelihood(x)

        expected = ((x - mean).square() / (-2 * var)).sum(dim=1).exp()
        expected = expected / (var * math.pi * 2).prod(dim=1).sqrt()
        expected = weights.mul(expected).sum().log().item()

        assert abs(log_likes - expected) < 1e-4, log_likes - expected

    def test_log_like_per_component(self):
        nmix = 8
        dim = 2

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        log_likes = torch.from_numpy(diag_gmm.log_likelihoods(x))

        expected = ((x - mean).square() / (-2 * var)).sum(dim=1).exp()
        expected = expected / (var * math.pi * 2).prod(dim=1).sqrt()
        expected = weights.mul(expected).log()

        assert torch.allclose(log_likes, expected), (log_likes - expected).max()

    def test_log_like_per_component_2d(self):
        nmix = 10
        dim = 3

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        N = 3
        x = torch.rand(N, dim)
        log_likes = torch.from_numpy(diag_gmm.log_likelihoods_matrix(x))
        assert log_likes.shape == (N, nmix), log_likes.shape

        expected_list = []
        for i in range(N):
            expected = ((x[i] - mean).square() / (-2 * var)).sum(dim=1).exp()
            expected = expected / (var * math.pi * 2).prod(dim=1).sqrt()
            expected = weights.mul(expected).log()
            expected_list.append(expected)
        expected = torch.stack(expected_list)

        assert torch.allclose(log_likes, expected), abs(log_likes - expected).max()

    def test_log_like_per_component_preselect(self):
        nmix = 10
        dim = 5

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        # indexes does not to be sorted or unique
        indexes = [0, 1, 3, 8, 7, 8, 3, 2]

        # log_likelihoods_preselect()
        # return loglike of selected components for a 1-D input x
        log_likes = torch.from_numpy(diag_gmm.log_likelihoods_preselect(x, indexes))
        assert log_likes.shape[0] == len(indexes)

        expected = ((x - mean).square() / (-2 * var)).sum(dim=1).exp()
        expected = expected / (var * math.pi * 2).prod(dim=1).sqrt()
        expected = weights.mul(expected).log()
        expected = expected[indexes]

        assert torch.allclose(log_likes, expected, atol=1e-4), (log_likes, expected)

    def test_gaussian_selection_1d(self):
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
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        # Select the top 3 components sorted by loglikes
        num_gselect = 3
        # log_like: a float, the total loglike of the selected components
        # indexes: the indexes of the selected components
        log_like, indexes = diag_gmm.gaussian_selection_1d(x, num_gselect)

        log_likes = torch.from_numpy(diag_gmm.log_likelihoods(x))
        sorted_log_likes, sorted2unsorted = torch.sort(log_likes, descending=True)

        assert indexes == sorted2unsorted[:num_gselect].tolist()
        assert abs(log_like - sorted_log_likes[:num_gselect].logsumexp(0).item()) < 1e-4

    def test_gaussian_selection_2d(self):
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
        diag_gmm.compute_gconsts()

        N = 5
        x = torch.rand(N, dim)
        # Select the top 3 components sorted by loglikes
        num_gselect = 3
        # log_like: a float, the total loglike of the selected components
        # indexes: the indexes of the selected components
        log_like, indexes_list = diag_gmm.gaussian_selection_2d(x, num_gselect)

        total = 0
        for i in range(N):
            log_like_i, indexes = diag_gmm.gaussian_selection_1d(x[i], num_gselect)
            assert indexes == indexes_list[i]
            total += log_like_i

        assert abs(log_like - total) < 1e-4, (log_like, total)

    def test_gaussian_selection_preselect(self):
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
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        # indexes does not to be sorted or unique
        indexes = [0, 1, 3, 8, 7, 8, 3, 2]
        num_gselect = 3

        # log_likelihoods_preselect()
        # return loglike of selected components for a 1-D input x
        log_like, selected_indexes = diag_gmm.gaussian_selection_preselect(
            x, preselect=indexes, num_gselect=num_gselect
        )

        log_likes = torch.from_numpy(diag_gmm.log_likelihoods_preselect(x, indexes))
        sorted_log_likes, sorted2unsorted = torch.sort(log_likes, descending=True)

        for i in range(num_gselect):
            assert selected_indexes[i] == indexes[sorted2unsorted[i]]

        assert abs(log_like - sorted_log_likes[:num_gselect].logsumexp(0).item()) < 1e-4

    def test_component_posteriors(self):
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
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)
        log_like, posteriors = diag_gmm.component_posteriors(x)

        posteriors = torch.from_numpy(posteriors)

        log_likes = torch.from_numpy(diag_gmm.log_likelihoods(x))

        assert torch.allclose(posteriors, log_likes.softmax(0))

        assert abs(log_like - log_likes.logsumexp(0).item()) < 1e-4

    def test_component_log_likelihood(self):
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
        diag_gmm.compute_gconsts()

        x = torch.rand(dim)

        log_likes = torch.from_numpy(diag_gmm.log_likelihoods(x))

        for i in range(nmix):
            log_like = diag_gmm.component_log_likelihood(x, i)
            assert abs(log_like - log_likes[i].item()) < 1e-4, (log_like, log_likes[i])

    def test_generate(self):
        nmix = 19
        dim = 10

        diag_gmm = khg.DiagGmm()
        diag_gmm.resize(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim) * 10
        var = torch.rand(nmix, dim) * 10
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        x = torch.from_numpy(diag_gmm.generate())

        assert x.shape[0] == dim, x.shape

        i = -1
        for k in range(nmix):
            if torch.all((mean[k] - 3 * var[k].sqrt()).le(x)) and torch.all(
                x.le(mean[k] + 3 * var[k].sqrt())
            ):
                i = k

        assert i != -1

    def test_perturb(self):
        nmix = 24
        dim = 10

        diag_gmm = khg.DiagGmm(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim) * 10
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        perturb_factor = 0.1
        diag_gmm.perturb(perturb_factor)
        for i in range(nmix):
            # perturb does not change the variance
            assert torch.allclose(torch.from_numpy(diag_gmm.vars[i]), var[i])

        # mean is changed to mean + perturb_factor * randn() * sqrt(var)

    def test_copy_from_diag_gmm(self):
        nmix = 24
        dim = 10

        diag_gmm = khg.DiagGmm(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim) * 10
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        dgm = khg.DiagGmm()
        dgm.copy_from_diag_gmm(diag_gmm)
        assert dgm.valid_gconsts == diag_gmm.valid_gconsts
        assert torch.allclose(
            torch.from_numpy(dgm.means), torch.from_numpy(diag_gmm.means)
        )
        assert torch.allclose(
            torch.from_numpy(dgm.vars), torch.from_numpy(diag_gmm.vars)
        )
        assert torch.allclose(
            torch.from_numpy(dgm.gconsts), torch.from_numpy(diag_gmm.gconsts)
        )

    def test_constructor_merge_diag_gmm(self):
        nmix = 24
        dim = 10

        diag_gmm = khg.DiagGmm(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        nmix2 = 30
        diag_gmm2 = khg.DiagGmm(nmix2, dim)

        weights2 = torch.rand(nmix2, dtype=torch.float32)
        weights2 /= weights2.sum()

        mean2 = torch.rand(nmix2, dim)
        var2 = torch.rand(nmix2, dim)
        diag_gmm2.set_weights(weights2)
        diag_gmm2.set_means(mean2)
        diag_gmm2.set_invvars(1 / var2)
        diag_gmm2.compute_gconsts()

        w = 0.2
        w2 = 0.8
        dgm = khg.DiagGmm([(w, diag_gmm), (w2, diag_gmm2)])

        assert dgm.num_gauss == nmix + nmix2
        assert dgm.dim == dim
        assert torch.allclose(torch.from_numpy(dgm.weights[:nmix] / w), weights)
        assert torch.allclose(torch.from_numpy(dgm.weights[nmix:] / w2), weights2)

        assert torch.allclose(torch.from_numpy(dgm.means[:nmix]), mean)
        assert torch.allclose(torch.from_numpy(dgm.vars[:nmix]), var)

        assert torch.allclose(torch.from_numpy(dgm.means[nmix:]), mean2)
        assert torch.allclose(torch.from_numpy(dgm.vars[nmix:]), var2)

    def test_interpolate(self):
        nmix = 24
        dim = 10

        diag_gmm = khg.DiagGmm(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.randn(nmix, dim)
        var = torch.randn(nmix, dim).square()
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        diag_gmm2 = khg.DiagGmm(nmix, dim)

        weights2 = torch.rand(nmix, dtype=torch.float32)
        weights2 /= weights2.sum()

        mean2 = torch.randn(nmix, dim)
        var2 = torch.randn(nmix, dim).square()
        diag_gmm2.set_weights(weights2)
        diag_gmm2.set_means(mean2)
        diag_gmm2.set_invvars(1 / var2)
        diag_gmm2.compute_gconsts()

        # update only weights
        rho = 0.1
        dgm = khg.DiagGmm(diag_gmm)
        # dgm = (1-rho)*dgm + rho*diag_gmm2
        dgm.interpolate(rho=rho, source=diag_gmm2, flags=khg.GmmUpdateFlags.kGmmWeights)
        assert torch.allclose(
            torch.from_numpy(dgm.weights),
            (1 - rho) * torch.from_numpy(diag_gmm.weights)
            + rho * torch.from_numpy(diag_gmm2.weights),
        )
        assert torch.allclose(
            torch.from_numpy(dgm.means), torch.from_numpy(diag_gmm.means)
        )

        assert torch.allclose(
            torch.from_numpy(dgm.vars), torch.from_numpy(diag_gmm.vars)
        )

        # update only means
        rho = 0.2
        dgm = khg.DiagGmm()
        dgm.copy_from_diag_gmm(diag_gmm)
        # dgm = (1-rho)*dgm + rho*diag_gmm2
        dgm.interpolate(rho=rho, source=diag_gmm2, flags=khg.GmmUpdateFlags.kGmmMeans)
        assert torch.allclose(
            torch.from_numpy(dgm.weights), torch.from_numpy(diag_gmm.weights)
        )
        assert torch.allclose(
            torch.from_numpy(dgm.means),
            (1 - rho) * torch.from_numpy(diag_gmm.means)
            + rho * torch.from_numpy(diag_gmm2.means),
        )
        assert torch.allclose(
            torch.from_numpy(dgm.vars), torch.from_numpy(diag_gmm.vars)
        )

        # update only variances
        rho = 0.3
        dgm = khg.DiagGmm(diag_gmm)
        # dgm = (1-rho)*dgm + rho*diag_gmm2
        dgm.interpolate(
            rho=rho,
            source=diag_gmm2,
            flags=khg.GmmUpdateFlags.kGmmVariances,
        )
        assert torch.allclose(
            torch.from_numpy(dgm.weights), torch.from_numpy(diag_gmm.weights)
        )
        assert torch.allclose(
            torch.from_numpy(dgm.means), torch.from_numpy(diag_gmm.means)
        )
        assert torch.allclose(
            torch.from_numpy(dgm.vars),
            (1 - rho) * torch.from_numpy(diag_gmm.vars)
            + rho * torch.from_numpy(diag_gmm2.vars),
        )

        # update all
        rho = 0.8
        dgm = khg.DiagGmm()
        dgm.copy_from_diag_gmm(diag_gmm)
        # dgm = (1-rho)*dgm + rho*diag_gmm2
        dgm.interpolate(
            rho=rho,
            source=diag_gmm2,
            flags=khg.GmmUpdateFlags.kGmmAll,
        )

        assert torch.allclose(
            torch.from_numpy(dgm.weights),
            (1 - rho) * torch.from_numpy(diag_gmm.weights)
            + rho * torch.from_numpy(diag_gmm2.weights),
        )

        assert torch.allclose(
            torch.from_numpy(dgm.means),
            (1 - rho) * torch.from_numpy(diag_gmm.means)
            + rho * torch.from_numpy(diag_gmm2.means),
        )

        assert torch.allclose(
            torch.from_numpy(dgm.vars),
            (1 - rho) * torch.from_numpy(diag_gmm.vars)
            + rho * torch.from_numpy(diag_gmm2.vars),
        )

    def test_pickle(self):
        nmix = 24
        dim = 10

        diag_gmm = khg.DiagGmm(nmix, dim)

        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.randn(nmix, dim)
        var = torch.randn(nmix, dim).square()
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)
        diag_gmm.compute_gconsts()

        data = pickle.dumps(diag_gmm, 2)  # Must use pickle protocol >= 2
        diag_gmm2 = pickle.loads(data)

        assert diag_gmm2.valid_gconsts is True
        assert torch.allclose(
            torch.from_numpy(diag_gmm2.weights), torch.from_numpy(diag_gmm2.weights)
        )
        assert torch.allclose(
            torch.from_numpy(diag_gmm2.means_invvars),
            torch.from_numpy(diag_gmm2.means_invvars),
        )
        assert torch.allclose(
            torch.from_numpy(diag_gmm2.inv_vars), torch.from_numpy(diag_gmm2.inv_vars)
        )


if __name__ == "__main__":
    torch.manual_seed(20230414)
    unittest.main()
