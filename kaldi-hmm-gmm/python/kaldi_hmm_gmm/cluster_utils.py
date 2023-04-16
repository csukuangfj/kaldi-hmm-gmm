import _kaldi_hmm_gmm as _khg


class RefineClustersOptions(object):
    def __init__(self, num_iters: int = 100, top_n: int = 5) -> None:
        """
        Args:
          num_iters:
            It must be >= 0. If zero, does nothing.
          top_n:
            It must be >= 2.
        """
        self.impl = _khg.RefineClustersOptions(
            num_iters=num_iters,
            top_n=top_n,
        )


class ClusterKMeansOptions(object):
    def __init__(
        self,
        refine_cfg: RefineClustersOptions,
        num_iters: int = 20,
        num_tries: int = 2,
        verbose: bool = true,
    ) -> None:
        self.impl = _khg.ClusterKMeansOptions(
            refine_cfg=refine_cfg.impl,
            num_iters=num_iter,
            num_tries=num_tries,
            verbose=verbose,
        )
