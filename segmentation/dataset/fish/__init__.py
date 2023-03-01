from .. import colors, CPARTS, DATASET_TYPES
from ..visualize_composite_labels import display_composite_annotations

dataset_splits = {"train": 0.85, "val": 0.05, "test": 0.1}
composite_labels = []

__all__ = ["composite_labels", "test_set_ratio", "visualize_composite_labels", "colors", "CPARTS", "DATASET_TYPES"]
