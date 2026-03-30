import pandas as pd


class DatasetCleaner:
    """
    Data Cleaning and Preprocessing Layer

    This class is responsible for transforming raw datasets into a clean,
    analysis-ready format. It encapsulates all data cleaning logic to ensure
    consistency, reusability, and separation from ingestion and modeling layers.

    Responsibilities:
    - Handle missing values (imputation or removal)
    - Remove duplicates
    - Standardize column names and data types
    - Detect and handle outliers (if required)
    - Apply basic data validation rules

    Attributes:
        df (pd.DataFrame): The input dataset to be cleaned

    Design Notes:
    - Cleaning methods should not mutate data silently; return cleaned copies where possible
    - Each transformation step should be modular (one responsibility per method)
    - Avoid mixing feature engineering with cleaning logic

    Example Workflow:
        cleaner = DatasetCleaner(df)
        df = cleaner.handle_missing_values()
        df = cleaner.remove_duplicates()
        df = cleaner.standardize_columns()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the DatasetCleaner.

        Args:
            df (pd.DataFrame): Raw dataset to be cleaned
        """
        self.df = df