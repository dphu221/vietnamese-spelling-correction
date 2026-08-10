"""Correction adapters and text-processing service."""

from .adapters import DemoCorrectionAdapter, UnavailableCorrectionAdapter
from .service import CorrectionService

__all__ = ["CorrectionService", "DemoCorrectionAdapter", "UnavailableCorrectionAdapter"]
