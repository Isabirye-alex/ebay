from typing import List, Dict, Any, Callable
from utils.logging import setup_logger
import pandas as pd
from utils.timing import timeit
from utils.validators import ReusableFunctions
from utils.exceptions import PipelineError, DataTypeError, DataValidationError


class FeatureEngineering:
    """Stateful Feature engineering class for ecommerce datasets

    This class performs step-by-setp feature engineering on self.df dataframe

    Each step is modular, logs its actions and updates quality metrics for tracking added columns

    Attributes:
        df (pd.Dataframe) : The input Dataframe being feature engineered
        logger(Logging.logger) : The logging instance for pipeline messages
        quality_metrics  (Dicy[str, Any]) : Tracks columns added and the execution status
        pipeline_steps (List[Callable]) : The ordered list of steps to be executed

    Raise:
        Key value error in case required column(s) are not found in the input dataset

    Outputs:
        Feature Engineered dataset (pd.DataFrame) : Fully featured dataframe with new added features derived from existing columns

    """

    REQUIRED_COLUMNS: List = [
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

    def __init__(self, dataframe):
        """
        Initialize FeatureEngineering to add new features (columns)

        Args:
            df (pd.DataFrame) : Dataset to be worked on
        """

        # Initialize dataset and create a copy of it
        self.df = dataframe.copy()

        # Initalize logger instance to track execution status
        self.logger = setup_logger(self.__class__.__name__)

        # Intialize Dataset validator to check whether the required columns exist with the input dataset
        self.validator = ReusableFunctions().validate_schema(self.df, self.REQUIRED_COLUMNS, self.logger)

        # Initialize a dictionary of quality metrics to store and keep track of data quality
        self.quality_metrics : Dict[str, Any] = {

        }

        # Initialize an iterable list of pipeline steps to be executed
        self.pipeline_steps : List[Callable[[], None]] = [
            self._compute_total_price

        ]

        # The _log_steps() function is to capture the impact of the feature engineering process and log whole process
    def _log_steps(self, step_name: str):
        '''
            Log the impact of feature engineering
            Args:
            step_name (str) : Name of the feature engineering step
        '''
        self.logger.info(f'{step_name}: ')
        
    def _compute_total_price(self):
        self.df['TotalPrice'] = self.df['price'] + self.df['shipping']

    @timeit 
    def run_features_pipeline(self)->pd.DataFrame:
        self.validator
        self.logger.info(f'Feature Engineering Started')

        for step in self.pipeline_steps:
            try:
                step()
            except Exception as e:
                raise RuntimeError(f'Fatal error occurred while adding features with error reason {e}')

        self.logger.info(f'Feature Engineering completed successfully')

        return self.df
