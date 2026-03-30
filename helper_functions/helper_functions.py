class HelperFunctions:
    """
    Utility and Resilience Layer

    This class groups reusable helper methods used across the pipeline,
    including wrappers for retry logic, error handling, and common utilities.

    Responsibilities:
    - Provide retry mechanisms for unstable operations (e.g., I/O, network, DB calls)
    - Standardize error handling and logging
    - Encapsulate reusable helper functions to avoid duplication

    Design Notes:
    - Methods in this class should be stateless and reusable
    - Prefer staticmethods or classmethods where instance state is not required
    - Avoid mixing business logic with utility logic

    Example Use Cases:
    - Retrying failed API/database requests
    - Wrapping functions with exponential backoff
    - Common data transformations or validations
    """

    def __init__(self) -> None:
        """Initialize helper utilities (no state required)."""
        pass
