"""Stitching subpackage for fragment candidate reassembly."""

from .overlap_dp import stitch_fragments
from .consistency import check_overlap_compatibility, compute_mismatch_penalty

__all__ = ["stitch_fragments", "check_overlap_compatibility", "compute_mismatch_penalty"]
