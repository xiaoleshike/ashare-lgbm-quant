"""Explicit exceptions for data ingestion failures."""

from __future__ import annotations


class DataIngestionError(Exception):
    """Base class for data ingestion errors."""


class TushareTokenError(DataIngestionError):
    """Raised when a required Tushare token is unavailable."""


class TushareRequestError(DataIngestionError):
    """Raised when a Tushare request fails after retries."""


class TusharePermissionError(TushareRequestError):
    """Raised when the account lacks permission for a Tushare endpoint."""


class TushareParameterError(TushareRequestError):
    """Raised when Tushare rejects deterministic request parameters."""


class TushareNoDataError(TushareRequestError):
    """Raised when Tushare reports that a valid request has no data."""


class DataValidationError(DataIngestionError):
    """Raised when downloaded or stored data violates expected constraints."""
