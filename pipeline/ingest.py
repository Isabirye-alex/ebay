import pandas as pd
from sqlalchemy import create_engine
from utils.timing import timeit
from utils.logging import setup_logger
from utils.retry import retry
from typing import Dict, Any, Callable
from utils.helpers import validate_file_path

logger = setup_logger()


class DataIngestor:
    """
    Data Ingestion and Persistence Layer

    This class is responsible for:
    - Loading raw data from external sources (e.g., CSV files)
    - Validating input data sources
    - Persisting processed or raw data into a PostgreSQL database

    Responsibilities:
    - Ensure data source availability before ingestion
    - Handle retries for unstable I/O operations
    - Provide a consistent interface for saving DataFrames to a database

    Attributes:
        source_path (str): Path to the input data source (e.g., CSV file)

    Notes:
        - Designed to be used as part of a larger data pipeline
        - All critical operations are wrapped with retry and timing decorators
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize the DataIngestor.

        Args:
            filepath (str): Path to the source CSV file
        """
        self.source_path = filepath
        self.quality_metrics: Dict[str, Any] = {
            "initial_rows": None,
            "source_path": self.source_path,
        }

    @timeit
    @retry(max_retries=3, delay=2)
    def fetch_raw_csv_data(self) -> Dict:
        """
        Load raw data from a CSV file.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If the file does not exist
            RuntimeError: If data loading fails after retries
        """
        try:
            validate_file_path(self.source_path)

            dataframe = pd.read_csv(self.source_path)
            total_rows = len(dataframe)
            shape = dataframe.shape
            self.quality_metrics["initial_rows"] = total_rows
            self.quality_metrics["Dataset Shape"] = shape
            logger.info(f"Data successfully loaded from {self.source_path}")
            logger.info(f"Loaded Dataset contains {total_rows} rows")
            logger.info(f"Loaded dataset has a shape of {shape}")

            return {"dataframe": dataframe, "quality_metrics": self.quality_metrics}

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(
                f"A fatal error occurred while fetching the dataset: {e}"
            )

    @timeit
    @retry(max_retries=3, delay=2)
    def save_to_postgres(
        self,
        df: pd.DataFrame,
        table_name: str,
        connection_string: str,
    ) -> None:
        """
        Persist a DataFrame to a PostgreSQL table.

        Converts Period and category columns to string before writing
        since PostgreSQL does not support these dtypes natively.

        Args:
            df (pd.DataFrame): Data to persist.
            table_name (str): Target table name.
            connection_string (str): SQLAlchemy connection string.

        Raises:
            RuntimeError: If the database write fails.
        """

        engine = create_engine(connection_string)

        try:
            df = df.copy()

            # PostgreSQL does not support Period dtype
            for col in df.columns:
                if pd.api.types.is_period_dtype(df[col]):  # type: ignore
                    df[col] = df[col].astype(str)

            # PostgreSQL does not support category dtype
            for col in df.select_dtypes(include=["category"]).columns:
                df[col] = df[col].astype(str)

            df.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False,
            )

            logger.info(f"Saved {len(df)} rows to table '{table_name}'")

        except Exception as e:
            logger.error(f"Failed to save '{table_name}' to PostgreSQL: {e}")
            raise RuntimeError(f"Database write failed for table '{table_name}'") from e
