import pandas as pd
from typing import Dict, Any, Callable, List
from utils.timing import timeit
from utils.logging import setup_logger
from utils.validators import ReusableFunctions
from utils.exceptions import PipelineError, DataValidationError, DataCleaningError, DataTypeError


class DatasetCleaner:
    """
    Stateful Data Cleaning Pipeline for eCommerce datasets.

    This class performs step-by-step cleaning of a DataFrame while mutating
    self.df in-place. Each step is modular, logs its actions, and updates
    quality metrics for tracking row removals and data issues.

    Attributes:
        df (pd.DataFrame): The input dataset being cleaned
        logger (logging.Logger): Logger instance for pipeline messages
        quality_metrics (Dict[str, Any]): Tracks row removals and cleaning info
        pipeline_steps (List[Callable]): Ordered list of cleaning functions
    """

    # Required columns in the dataset
    REQUIRED_COLUMNS = [
        "category",
        "title",
        "price",
        "condition",
        "sold_date",
        "shipping",
        "rating",
        "reviews_count",
        "seller_feedback_percentage",
        "seller_feedback_rating",
        "seller_feedback_count",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize DatasetCleaner.

        Args:
            df (pd.DataFrame): Raw dataset to be cleaned
        """
        # Work on a copy to avoid accidental external mutation
        self.df = df.copy()

        # Set up logger for this class
        self.logger = setup_logger(self.__class__.__name__)

        # Initialize metrics dictionary
        self.quality_metrics: Dict[str, Any] = {
            "initial_rows": len(self.df),
            "rows_removed_total": 0,
        }

        # Validator injection
        self.schema_validator = ReusableFunctions().validate_schema(self.df, self.REQUIRED_COLUMNS, self.logger)

        # Ordered cleaning steps
        self.pipeline_steps: List[Callable] = [
            self._clean_category,
            self._clean_title,
            self._clean_price,
            self._clean_condition,
            self._clean_date,
            self._clean_shipping,
            self._clean_rating,
            self._clean_reviews_count,
            self._clean_seller_feedback_count,
            self._clean_seller_feedback_rating,
            self._clean_seller_feedback_percentage,
            self._drop_url,
            self._optimize_columns
        ]

    # INTERNAL UTILITIES

    def _log_step(self, step_name: str, before: int, after: int):
        """
        Log the impact of a cleaning step and update total rows removed.

        Args:
            step_name (str): Name of the cleaning step
            before (int): Row count before the step
            after (int): Row count after the step
        """
        removed = before - after
        self.quality_metrics["rows_removed_total"] += removed
        self.logger.info(f"{step_name}: removed {removed} rows")

    # CLEANING STEPS

    def _clean_category(self):
        """Drop rows with missing 'category'."""
        before = len(self.df)
        try:
            self.df = self.df.dropna(subset=["category"])
            after = len(self.df)
            self.quality_metrics["category_rows_removed"] = before - after
            self._log_step("category_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_title(self):
        """Drop rows with missing 'title'."""
        before = len(self.df)
        try:
            self.df['title'] = self.df['title'].str.strip().str.lower().astype(str)
            self.df = self.df.dropna(subset=["title"])
            after = len(self.df)
            self.quality_metrics["title_rows_removed"] = before - after
            self._log_step("title_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_price(self):
        """Clean 'price' column: remove symbols, convert to float, drop invalid rows."""
        before = len(self.df)
        try:
            # Drop rows where price is missing
            self.df = self.df.dropna(subset=["price"])

            # Remove $ and commas, strip whitespace
            self.df["price"] = (
                self.df["price"].str.replace(r"[\$,]", "", regex=True).str.strip()
            )

            # Convert to numeric, coerce errors to NaN
            self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")

            # Drop rows where conversion failed
            self.df = self.df.dropna(subset=["price"])

            after = len(self.df)
            self.quality_metrics["price_rows_removed"] = before - after
            self._log_step("price_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_condition(self):
        """Drop rows with missing 'condition'."""
        before = len(self.df)
        try:
            self.df["condition"] = self.df["condition"].fillna("unkown")
            self.df['condition'] = self.df['condition'].str.strip().str.lower().astype(str)
            after = len(self.df)
            self.quality_metrics["condition_rows_removed"] = before - after
            self._log_step("condition_cleaning", before, after)
        except Exception as e:
             raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_date(self):
        """Convert 'sold_date' to datetime and drop invalid rows."""
        before = len(self.df)
        try:
            self.df["sold_date"] = pd.to_datetime(self.df["sold_date"], errors="coerce")
            self.df = self.df.dropna(subset=["sold_date"])
            after = len(self.df)
            self.quality_metrics["date_rows_removed"] = before - after
            self._log_step("date_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_shipping(self):
        """Clean 'shipping' column: remove symbols, convert to float, impute missing values."""
        before = len(self.df)
        try:
            # Remove $ and unwanted text
            self.df["shipping"] = (
                self.df["shipping"]
                .str.replace(r"\$", "", regex=True)
                .str.replace("delivery", "", regex=False)
            )

            # Convert to numeric
            self.df["shipping"] = pd.to_numeric(self.df["shipping"], errors="coerce")

            # Track missing values before filling
            missing = self.df["shipping"].isna().sum()
            self.quality_metrics["shipping_missing_before_fill"] = missing
            self.logger.info(
                f"Missing rows in Shipping column before filling were {missing}"
            )

            # Fill missing shipping with mean
            self.df["shipping"] = self.df["shipping"].fillna(0)

            after = len(self.df)
            self._log_step("shipping_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_reviews_count(self):
        """Handle missing reviews_count intelligently."""
        before = len(self.df)
        try:
            # Flag missing values
            self.df["reviews_count_missing"] = self.df["reviews_count"].isna().astype(int)

            # Fill missing as 0 (no reviews)
            self.df["reviews_count"] = self.df["reviews_count"].fillna(0)

            after = len(self.df)
            self._log_step("reviews_count_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_rating(self):
        """
        Handle missing rating values.
        Creates a binary flag for missingness before imputing with median.
        Median is used over mean because eBay ratings are right-skewed.
        """
        before = len(self.df)
        try:
            self.df['rating_missing'] = self.df['rating'].isna().astype(int)
            self.df['rating']         = self.df['rating'].fillna(self.df['rating'].median())

            after = len(self.df)
            self._log_step("rating_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred with error reason {e}')

    def _clean_seller_feedback_count(self):
        """Handle missing 'seller_feedback_count' column."""
        before = len(self.df)
        try:
            self.df["seller_feedback_count"] = pd.to_numeric(
                self.df["seller_feedback_count"], errors="coerce"
            )

            self.df["seller_feedback_count_missing"] = (
                self.df["seller_feedback_count"].isna().astype(int)
            )

            self.df["seller_feedback_count"] = self.df["seller_feedback_count"].fillna(0)
            after = len(self.df)  
            self._log_step("seller_feedback_count_cleaning", before, after) 
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred while cleaning the seller feedback count column with error reason {e}')


    def _clean_seller_feedback_rating(self):
        """Clean categorical seller feedback rating."""
        before = len(self.df)
        try:
            # Normalize text (lowercase, strip spaces)
            self.df['seller_feedback_rating'] = (
                self.df['seller_feedback_rating']
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # Flag missing
            self.df['seller_feedback_rating_missing'] = (
                self.df['seller_feedback_rating'].isna().astype(int)
            )

            # Fill missing with 'unknown'
            self.df['seller_feedback_rating'] = self.df[
                'seller_feedback_rating'
            ].replace('nan', pd.NA).fillna('unknown')

            after = len(self.df)
            self._log_step("seller_feedback_rating_cleaning", before, after)
        except Exception as e:
            raise DataCleaningError(f'Fatal error occurred while cleaning the seller feedback rating with error reason {e}')

    def _clean_seller_feedback_percentage(self):
        """Handle missing 'seller_feedback_percentage' column."""
        before = len(self.df)
        try: 
            self.df["seller_feedback_percentage"] = (
                self.df["seller_feedback_percentage"]
                .astype(str)
                .str.replace("%", "", regex=False)
            )

            self.df["seller_feedback_percentage"] = pd.to_numeric(
                self.df["seller_feedback_percentage"], errors="coerce"
            )

            self.df["seller_feedback_percentage_missing"] = (
                self.df["seller_feedback_percentage"].isna().astype(int)
            )

            self.df["seller_feedback_percentage"] = self.df[
                "seller_feedback_percentage"
            ].fillna(self.df["seller_feedback_percentage"].median())

            after = len(self.df)
            self._log_step("seller_feedback_percentage_cleaning", before, after)
        except Exception as e: 
            raise DataCleaningError(f'Fatal error occurred while cleaning seller feedback percentage column with error reason {e}')


    def _optimize_columns(self):
        cat_cols = ['category', 'condition', 'seller_feedback_rating', 'title']

        for c in cat_cols:
            self.df[c] = self.df[c].astype('category')

    def _drop_url(self):
        self.df = self.df.drop(columns='url')

    # PIPELINE EXECUTION
    @timeit
    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline sequentially.

        Returns:
            pd.DataFrame: The fully cleaned dataset
        """
        self.logger.info("Starting data cleaning pipeline")
        # Validate the dataset
        self.schema_validator
        # Execute each cleaning step
        for step in self.pipeline_steps:
            try:
             step()
            except PipelineError as e:
                self.logger.error(f'Pipeline Failed with error reason {e}')
                raise

            except Exception as e:
                self.logger.error('Unexpected error occurred with reason {e}')
                raise
        self.logger.info("Pipeline completed successfully")
        self.logger.info(f"Final row count: {len(self.df)}")
        self.logger.info(f"Quality metrics: {self.quality_metrics}")

        return self.df
