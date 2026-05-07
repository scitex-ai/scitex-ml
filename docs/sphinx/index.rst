SciTeX ML
=========

**scitex-ml** is the standalone home of classical / deep machine-learning
utilities — classification reporters, time-series CV, training helpers,
loss / metrics / optim, clustering, sampling, and sklearn integration.

.. code-block:: python

   import scitex_ml

   clf = scitex_ml.Classifier("LogisticRegression")
   clf.fit(X_tr, y_tr)
   reporter = scitex_ml.ClassificationReporter("./results")
   reporter.calc_metrics(y_te, clf.predict(X_te), clf.predict_proba(X_te))

The umbrella ``scitex-python`` exposes this package as ``scitex.ml``.
Generative-AI utilities (``GenAI``, agents, image / audio / video) live in
the sibling package `scitex-genai
<https://github.com/ywatanabe1989/scitex-genai>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
