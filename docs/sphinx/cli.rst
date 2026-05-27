CLI Reference
=============

``scitex-ml`` exposes the stateless analysis surface from the terminal. Run
``scitex-ml --help`` for the live command list, or ``scitex-ml
--help-recursive`` to dump help for every subcommand at once.

.. code-block:: bash

   scitex-ml --version            # scitex-ml/X.Y.Z
   scitex-ml --help-recursive     # all commands + subcommands

Commands
--------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Category
     - Command
     - Description
   * - **Analysis**
     - ``scitex-ml compute-metrics``
     - Balanced accuracy, MCC, confusion matrix, ROC/PR-AUC from a predictions table
   * - **Analysis**
     - ``scitex-ml generate-report``
     - Full ClassificationReporter report (summary + plots) into a directory
   * - **Analysis**
     - ``scitex-ml reduce-dimensions``
     - PCA / UMAP 2-D projection figure from a feature matrix
   * - **Introspection**
     - ``scitex-ml list-python-apis``
     - List the public Python API (``-v`` signatures, ``-vv`` docstrings, ``--json``)
   * - **Integration**
     - ``scitex-ml mcp``
     - MCP server management (``start``, ``doctor``, ``list-tools``, ``install``)
   * - **Integration**
     - ``scitex-ml skills``
     - List / get / install the bundled agent-facing skill pages
   * - **Utility**
     - ``scitex-ml install-shell-completion``
     - Wire up ``<TAB>`` completion in the user's shell rc

Analysis verbs
--------------

Each verb is file-in → JSON/artifact-out and mirrors an MCP tool exactly (same
name, same JSON — see :doc:`mcp`).

.. code-block:: bash

   # Metrics from a saved predictions table (columns: y_true, y_pred[, y_proba])
   scitex-ml compute-metrics preds.csv
   scitex-ml compute-metrics preds.csv --label neg --label pos --json

   # Full report (summary.json + plots) under ./report
   scitex-ml generate-report preds.csv -o ./report

   # PCA / UMAP figure coloured by a label column
   scitex-ml reduce-dimensions features.csv -o pca.png --label-col target
   scitex-ml reduce-dimensions features.csv -o umap.png --method umap --supervised

Introspection
-------------

.. code-block:: bash

   scitex-ml list-python-apis          # names
   scitex-ml list-python-apis -vv      # + signatures + docstrings
   scitex-ml list-python-apis --json

Shell completion
----------------

.. code-block:: bash

   scitex-ml install-shell-completion --shell bash
   eval "$(scitex-ml print-shell-completion --shell bash)"
