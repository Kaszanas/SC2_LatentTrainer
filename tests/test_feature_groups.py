"""Tests for latent_trainer.features.feature_groups."""

from latent_trainer.features.feature_groups import FEATURE_GROUPS, NUM_GROUPS


class TestFeatureGroups:
    def test_count(self):
        assert NUM_GROUPS == 8
        assert len(FEATURE_GROUPS) == NUM_GROUPS

    def test_groups_are_contiguous(self):
        """Feature groups should tile [0, 203) with no gaps or overlaps."""
        prev_end = 0
        for name, start, end in FEATURE_GROUPS:
            assert start == prev_end, f"Gap or overlap before {name}: expected {prev_end}, got {start}"
            assert end > start, f"{name} has non-positive width"
            prev_end = end

    def test_total_features(self):
        """All groups should cover exactly 203 features."""
        total = sum(end - start for _, start, end in FEATURE_GROUPS)
        assert total == 203

    def test_names_are_unique(self):
        names = [name for name, _, _ in FEATURE_GROUPS]
        assert len(names) == len(set(names))

    def test_groups_are_tuples_of_correct_type(self):
        for item in FEATURE_GROUPS:
            name, start, end = item
            assert isinstance(name, str)
            assert isinstance(start, int)
            assert isinstance(end, int)
