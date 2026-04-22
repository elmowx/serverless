"""
Named Policy presets used as sanity checkpoints and baseline comparison
points in benchmark runs.
"""

from __future__ import annotations

from .types import Policy

BASELINES: dict[str, Policy] = {
    "aggressive": Policy(keep_alive_s=30.0, max_containers=20, prewarm_threshold=1.0),
    "balanced": Policy(keep_alive_s=300.0, max_containers=10, prewarm_threshold=1.0),
    "generous": Policy(keep_alive_s=1800.0, max_containers=10, prewarm_threshold=1.0),
    "prewarm_heavy": Policy(keep_alive_s=600.0, max_containers=10, prewarm_threshold=0.6),
    "minimal": Policy(keep_alive_s=60.0, max_containers=3, prewarm_threshold=1.0),
}
