"""
Surrogate models for Bayesian Optimization.

Five models are planned (all with fit / predict interface for bayesopt.loop):

1. GP (gp.py) — Gaussian Process, kernel-based, strong uncertainty; baseline.
2. BNN-MC (bnn_mc_dropout.py) — Bayesian NN with MC Dropout; flexible alternative.
3. PFN-BNN (pfn_bnn.py) — Prior Fitted Network; amortized Bayesian inference, in-context.
4. PFN-HEBO+ (pfn_hebo.py) — PFN with HEBO-inspired prior; robust to irrelevant dimensions.
5. GMM-NP (gmm_np.py) — Gaussian Mixture Neural Process; meta-learning, mixture posterior.
"""

