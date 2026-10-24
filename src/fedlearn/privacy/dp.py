"""fedlearn.privacy.dp — client-side differential privacy for FedAvg.

Mechanism (flat, per-client, per-round):
  1. L2-clip the raw weight update to norm ≤ C;
  2. add i.i.d. Gaussian noise with std σ = C · z (z = noise multiplier);
  3. the server averages the sanitized updates as usual.

The ε estimate below uses the analytic Gaussian mechanism with *basic*
composition over T rounds — an honest upper bound, clearly labeled as such in
the logs (tight RDP accounting is out of core scope; the numbers are for
the privacy-audit story, before/after). Requires numpy only.
"""

from __future__ import annotations

import math

import numpy as np

from ..log import get_logger

log = get_logger("privacy.dp")


class DPSettings:
    def __init__(self, enabled: bool = False, clip_C: float = 1.0,
                 noise_multiplier: float = 1.1, delta: float = 1e-5) -> None:
        self.enabled = bool(enabled)
        self.clip_C = float(clip_C)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)

    # -- the two operations applied to a flat weight update -------------------
    def clip(self, update: np.ndarray) -> tuple[np.ndarray, float]:
        norm = float(np.linalg.norm(update))
        if norm == 0.0:
            return update, 1.0
        scale = min(1.0, self.clip_C / norm)
        return update * scale, min(1.0, norm / self.clip_C + 1e-12)

    def sanitize(self, update: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Clip then add Gaussian noise with std = C·z. Returns the sanitized update."""
        if not self.enabled:
            return update
        clipped, ratio = self.clip(update)
        if rng is None:
            rng = np.random.default_rng()
        sigma = self.clip_C * self.noise_multiplier
        noise = rng.normal(0.0, sigma, size=clipped.shape).astype(np.float32)
        if ratio > 1.0 - 1e-9:
            log.debug(f"  dp: update UNCLIPPED (norm {ratio:.3f} ≤ C={self.clip_C})")
        else:
            log.debug(f"  dp: update clipped at C={self.clip_C} (raw ratio {ratio:.3f})")
        return (clipped + noise).astype(np.float32)

    # -- accounting (approximate, basic composition) --------------------------
    # ponytail: an honest upper bound beats a fake precise one; upgrade path =
    # RDP accountant (opacus) when tight ε matters.
    def estimate_epsilon(self, rounds: int) -> float:
        """Upper-bound ε for T rounds of the Gaussian mechanism (basic composition).

        Per-round ε₀ from the analytic Gaussian mechanism (δ fixed); total =
        T · ε₀. Loose, but honest — logs always say 'upper bound, basic
        composition'. Tight RDP accounting is a documented stretch.
        """
        if self.noise_multiplier <= 0:
            return math.inf
        # analytic Gaussian: ε₀ ≈ sqrt(2 * ln(1.25/δ)) / σ, σ = z
        per_round = math.sqrt(2 * math.log(1.25 / self.delta)) / self.noise_multiplier
        return rounds * per_round


def log_privacy_config(settings: DPSettings, rounds: int) -> None:
    if not settings.enabled:
        log.info("  dp: OFF — model is trainable but NOT differentially private")
        return
    eps = settings.estimate_epsilon(rounds)
    log.info(
        f"  dp: ON — C={settings.clip_C}, z={settings.noise_multiplier}, "
        f"T={rounds} rounds → ε ≤ {eps:.1f} (δ={settings.delta:.0e}, "
        f"UPPER BOUND, basic composition)"
    )
