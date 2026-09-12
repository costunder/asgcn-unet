"""Standard-library-only validation for the explicit v4 reconstruction design."""

import math


def validate_hierarchy_config(value, depth):
    required = {"after_layer", "spatial_cell_pixels", "temporal_cell_seconds", "edge_pseudo"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("v4 requires explicit after_layer, spatial_cell_pixels, temporal_cell_seconds, edge_pseudo")
    if type(value["after_layer"]) is not int or not 1 <= value["after_layer"] < depth:
        raise ValueError("Pooling must have trained graph convolutions both before and after it")
    for key in ("spatial_cell_pixels", "temporal_cell_seconds"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise ValueError(f"hierarchy_config.{key} must be positive and finite")
    if value["edge_pseudo"] != "mean_fine_distance_over_radius":
        raise ValueError("Coarse edges require the declared bounded fine-edge mean pseudo coordinate")
    return dict(value)
