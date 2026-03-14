"""SC2 feature group definitions.

Defines the semantic grouping of the 203-feature vector produced by
``rich_transform.py``.  These groups are used by:

- ``train_transformer.py`` (feature tokeniser)
- ``analyze_overfitting.py`` (feature ablation)
- ``visualize_attention.py`` (attention analysis)

Keeping the definitions here (close to the feature engineering code)
avoids the inverted dependency where analysis scripts import constants
from training scripts.
"""

from __future__ import annotations

FEATURE_GROUPS: list[tuple[str, int, int]] = [
    ("early_economy", 0, 39),
    ("mid_economy", 39, 78),
    ("late_economy", 78, 117),
    ("final_state", 117, 156),
    ("economy_delta", 156, 195),
    ("meta_stats", 195, 199),  # APM, MMR, SQ, supplyCappedPercent
    ("unit_activity", 199, 201),  # units_born, units_killed
    ("game_info", 201, 203),  # upgrades, duration
]

NUM_GROUPS: int = len(FEATURE_GROUPS)
