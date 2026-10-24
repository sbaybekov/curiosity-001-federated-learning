"""fedlearn.models — per-endpoint linear heads with clean parameter hooks.

Design (from the plan): NO shared trunk for GLMs. One independent linear
model per endpoint (logistic for cls, linear for reg), each with
``get_parameters()`` / ``set_parameters()`` hooks so Flower (or the manual
FedAvg loop) can exchange weights as flat arrays. Local training uses
scikit-learn SGD warm-continuation (partial_fit), which is what makes FedAvg
iterative instead of a single-shot refit.
"""

from .heads import LinearHead, init_heads, pack_heads, unpack_heads  # noqa: F401
