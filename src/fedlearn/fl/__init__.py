"""fedlearn.fl — the federated loop.

Two interchangeable backends share the exact same parameter convention:

* ``flower`` (primary, plan t11) — Flower ``NumPyClient`` + ``FedAvg``
  strategy via in-process simulation;
* ``manual`` (fallback) — a ~40-line FedAvg loop with identical math, used if
  flwr or ray misbehaves on the host.

Both report per-round progress; the logs say which backend actually ran.
"""

from . import client, server  # noqa: F401
