class PipelineError(Exception):
    """Base class for all pipeline-related errors."""

    pass


class DataValidationError(PipelineError):
    """Raised when dataset schema or structure is invalid."""

    pass


class DataCleaningError(PipelineError):
    """Raised when a cleaning step fails."""

    pass


class DataTypeError(PipelineError):
    """Raised when a column has unexpected data types."""

    pass
