"""Dataset ingestion and validation interfaces."""

from .brec import BRECDataset, BRECPair, download_brec, summarize_brec, validate_brec

__all__ = ["BRECDataset", "BRECPair", "download_brec", "summarize_brec", "validate_brec"]
