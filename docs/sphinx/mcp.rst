MCP Server
==========

scitex-ml ships a `Model Context Protocol <https://modelcontextprotocol.io/>`_
(MCP) server so AI agents can compute metrics, generate reports, and project
feature matrices autonomously — the same stateless analysis surface as the CLI.

Installation
------------

.. code-block:: bash

   pip install scitex-ml[mcp]

Starting the server
-------------------

.. code-block:: bash

   scitex-ml mcp start            # stdio transport
   scitex-ml mcp doctor           # health check (deps, tool count)

The scitex umbrella also mounts this server automatically under the ``ml``
namespace, so the tools surface to agents as ``ml_*``.

MCP host configuration
----------------------

.. code-block:: json

   {
     "mcpServers": {
       "scitex-ml": {
         "command": "scitex-ml",
         "args": ["mcp", "start"]
       }
     }
   }

Generate this snippet with ``scitex-ml mcp install``.

Available tools
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Tool (umbrella-mounted)
     - Description
   * - ``ml_compute_metrics``
     - Balanced accuracy, MCC, confusion matrix, ROC/PR-AUC from a predictions table
   * - ``ml_generate_report``
     - Full ClassificationReporter report (summary.json + plots) into a directory
   * - ``ml_reduce_dimensions``
     - PCA / UMAP 2-D projection figure from a feature matrix
   * - ``ml_skills_list`` / ``ml_skills_get``
     - Discover and fetch the skill pages shipped with scitex-ml

All tools take/return JSON. The analysis tools report input errors as
``{"success": false, "error": "..."}`` rather than raising.

Parity
------

Each tool maps 1:1 to a ``scitex-ml`` CLI verb with the same name and JSON
shape (``compute-metrics`` ↔ ``ml_compute_metrics``). scitex-ml deliberately
exposes only this stateless slice via MCP — training, optimizers and the deep
submodule API stay Python-only — so it declares ``mcp_parity_exempt = true``.

Diagnostics
-----------

.. code-block:: bash

   scitex-ml mcp doctor
   scitex-ml mcp list-tools -vv
