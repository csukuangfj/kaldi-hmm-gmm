DiagGmm
=======

gconsts
-------

The density of a diagonal-covariance Gaussian factorizes per dimension:

.. math::

	f_x(x_1, \ldots, x_k) &= \frac{\exp \Big(-\frac{1}{2} ( \frac{y_1^2}{\sigma_1^2} + \frac{y_2^2}{\sigma_2^2} + \ldots + \frac{y_k^2}{\sigma_k^2} ) \Big)}{\sqrt{(2\pi)^{k}\sigma_1^2 \cdot \sigma_2^2 \cdots \sigma_k^2}} \\
	&= \frac{\exp \Big(-\frac{1}{2} ( \frac{y_1^2}{\sigma_1^2} + \frac{y_2^2}{\sigma_2^2} + \ldots + \frac{y_k^2}{\sigma_k^2} ) \Big)}{\sqrt{2\pi\sigma_1^2 \cdot 2\pi\sigma_2^2 \cdots 2\pi\sigma_k^2}} \\
	&= \frac{\exp\Big(-\frac{y_1^2}{2\sigma_1^2} \Big)}{\sqrt{2\pi\sigma_1^2}}  \cdot \frac{\exp\Big(-\frac{y_2^2}{2\sigma_2^2} \Big)}{\sqrt{2\pi\sigma_2^2}}  \cdots \frac{\exp\Big(-\frac{y_k^2}{2\sigma_k^2} \Big)}{\sqrt{2\pi\sigma_k^2}} \\
	&= \frac{\exp\Big(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2} \Big)}{\sqrt{2\pi\sigma_1^2}}  \cdot \frac{\exp\Big(-\frac{(x_2-\mu_2)^2}{2\sigma_2^2} \Big)}{\sqrt{2\pi\sigma_2^2}}  \cdots \frac{\exp\Big(-\frac{(x_k-\mu_k)^2}{2\sigma_k^2} \Big)}{\sqrt{2\pi\sigma_k^2}} \\
	&= f_1(x_1) \cdot f_2(x_2) \cdots f_k(x_k)

.. math::

   \log f(x) = -0.5 \sum_{i=1}^{k}\left(\log 2\pi + \log \sigma_i^2 + \frac{(x_i-\mu_i)^2}{2\sigma_i^2}  \right)

We define ``gconst`` as the x-independent part of the mixture component's
log-likelihood (the log density at ``x=0`` plus the log weight):

.. math::

   \mathrm{gconst} = \log \lambda - 0.5 \left(k \log 2\pi + \sum_{i=1}^k \log \sigma_i^2 + \sum_{i=1}^k\frac{\mu_i^2}{\sigma_i^2}\right)

where :math:`\lambda` is the mixture weight, while :math:`k` is the dimension.
This is what :meth:`kaldi_hmm_gmm.DiagGmm.compute_gconsts` precomputes.

split
-----

How to increase the number of mixes?  This is kaldi's ``DiagGmm::Split``.

Find the component that has the largest weight.  Assume there is only one
component and we want to split it into two components.  In this case,
component 0 has the largest weight since there is only one component.

For the weight, we split it into two uniformly:

.. math::

   w_{0}^{new} = \frac{w_{0}}{2}\\
   w_{1}^{new} = \frac{w_{0}}{2}

For the standard deviation, we reuse it:

.. math::

   \sigma_{0}^{new} = \sigma_0\\
   \sigma_{1}^{new} = \sigma_0

For the mean, we use:

.. math::

   \mu_0^{new} = \mu_0 + \mathrm{perturb\_factor} \cdot r \cdot \sigma_0\\
   \mu_1^{new} = \mu_0 - \mathrm{perturb\_factor} \cdot r \cdot \sigma_0

where :math:`r` is a **standard Gaussian** random number (mean 0, variance 1;
kaldi's ``RandGauss()``) and ``perturb_factor`` is a user provided argument,
e.g., ``0.01``.

logdet
------

``logdet`` is defined as:

.. math::

   \mathrm{logdet} = \log \left( \frac{1}{\sigma_1^2 \sigma_2^2 ... \sigma_n^2} \right)^{1/2}

i.e. :math:`-\frac{1}{2}\sum_i \log \sigma_i^2`.

merge
-----

How to select two components to merge?  This is kaldi's ``DiagGmm::Merge``.

Suppose we have 3 components ``0``, ``1``, and ``2``; we compute the
objective improvement ``delta`` for every pair and merge the best one.  For
a pair (0, 1), the merged statistics are moment-matched:

.. math::

   \mu_{01} = \frac{w_0 \mu_0 + w_1 \mu_1}{w_0 + w_1}\\

.. math::

   \mathrm{var}_{01} = \frac{w_0 (\mathrm{var}_0 + \mu_0^2) + w_1 (\mathrm{var}_1 + \mu_1^2)}{w_0 + w_1} - \mu_{01}^2

The merged weight is ``w_0 + w_1``.

.. math::

   \mathrm{delta}\ \mathrm{loglike}_{01} = (w_0 + w_1) \cdot \mathrm{logdet}_{01} - w_0 \,\mathrm{logdet}_0 - w_1 \,\mathrm{logdet}_1

We want to **maximize this improvement**, so we choose the pair with the
largest ``delta loglike`` to merge.
