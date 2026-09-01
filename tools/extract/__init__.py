"""Structured web extraction utilities."""

from .scoring import CandidateScorer, create_candidate_scorer
from .webex import StructuredDataExtractor

__all__ = ["CandidateScorer", "StructuredDataExtractor", "create_candidate_scorer"]
