DiagGmm
=======

gconsts
-------

.. math::

	f_x(x_1, \ldots, x_k) &= \frac{\exp \Big(-\frac{1}{2} ( \frac{y_1^2}{\sigma_1^2} + \frac{y_2^2}{\sigma_2^2} + \ldots + \frac{y_k^2}{\sigma_k^2} ) \Big)}{\sqrt{(2\pi)^{k}\sigma_1^2 \cdot \sigma_2^2 \cdots \sigma_k^2}} \\
	&= \frac{\exp \Big(-\frac{1}{2} ( \frac{y_1^2}{\sigma_1^2} + \frac{y_2^2}{\sigma_2^2} + \ldots + \frac{y_k^2}{\sigma_k^2} ) \Big)}{\sqrt{2\pi\sigma_1^2 \cdot 2\pi\sigma_2^2 \cdots 2\pi\sigma_k^2}} \\
	&= \frac{\exp\Big(-\frac{y_1^2}{2\sigma_1^2} \Big)}{\sqrt{2\pi\sigma_1^2}}  \cdot \frac{\exp\Big(-\frac{y_2^2}{2\sigma_2^2} \Big)}{\sqrt{2\pi\sigma_2^2}}  \cdots \frac{\exp\Big(-\frac{y_k^2}{2\sigma_k^2} \Big)}{\sqrt{2\pi\sigma_k^2}} \\
	&= \frac{\exp\Big(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2} \Big)}{\sqrt{2\pi\sigma_1^2}}  \cdot \frac{\exp\Big(-\frac{(x_2-\mu_2)^2}{2\sigma_2^2} \Big)}{\sqrt{2\pi\sigma_2^2}}  \cdots \frac{\exp\Big(-\frac{(x_k-\mu_k)^2}{2\sigma_k^2} \Big)}{\sqrt{2\pi\sigma_k^2}} \\
	&= f_1(x_1) \cdot f_2(x_2) \cdots f_k(x_k)

.. math::

   \log f(x) = -0.5 \sum_{i=1}^{k}\left(\log 2\pi + \log \sigma_1^2 + \frac{(x_i-\mu_i)^2}{2\sigma_i^2}  \right)

We define ``gconst`` by setting ``x=0``:

.. math::

   \mathrm{gconst} = \log \lambda - 0.5 \left(k \log 2\pi + \sum_{i=1}^k \sigma_i^2 + \sum_{i=1}^k\frac{\mu_i^2}{\sigma_i^2}\right)


where :math:`\lambda` is the mixture weight, while :math:`k` is the dimension.

split
-----

How to increase the number of mixes?

Find the component that has the largest weight. Assume there is only one component
and we want to split it into two components. In this case, component 0 has the largest
weight since there is only one component.

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

   \mu_0^{new} = \mu_0 + \mathrm{perturb\_factor} \cdot \mathrm{rand()} \cdot \sigma_0\\
   \mu_1^{new} = \mu_0 - \mathrm{perturb\_factor} \cdot \mathrm{rand()} \cdot \sigma_0

where ``rand()`` returns a random number between ``0`` and ``1`` and ``perturb_factor``
is a user provided argument, e.g., ``0.01``.
