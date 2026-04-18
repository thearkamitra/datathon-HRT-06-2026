"""Linear probabilistic / heteroskedastic return model (Phase-2 method).

Implements the plan in ``plans/probablisticReg.txt``:

* a conditional-mean head (ridge over OHLC + news-as-identity features),
* a conditional-variance head fitted on OOF-residual magnitudes,
* optional quantile heads at q10/q50/q90,
* Gaussian projection of (mu, sigma) into the (mu, p_up, u) contract used
  by the shared Sharpe-aware sizer.

The module deliberately reuses three building blocks that already exist and
have been validated in this repository:

* OHLC session-level feature builder (``src/tailored-modeler/features.py``),
* first-half news featurizer (``src/regime/news.py``), and
* Sharpe-aware position sizer (``src/tailored-modeler/sizing.py``).

Those three are loaded via :mod:`importlib` from their sibling directories
inside :mod:`pipeline` so we do not duplicate hundreds of lines of tested
code while keeping this new method isolated in its own namespace.
"""
