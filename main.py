"""
Main Pipeline Entry Point

This script serves as the central execution point for the entire data pipeline.
It orchestrates the workflow from data ingestion to final output generation.

Responsibilities:
- Load the dataset from the specified file path
- Trigger preprocessing and cleaning steps
- Execute feature engineering (if applicable)
- Run analysis or model training
- Save outputs (processed data, models, or reports)

Variables:
- file_path (str): Path to the input dataset used throughout the pipeline

Usage:
    Run this script directly to execute the full pipeline:
        python main.py

Notes:
- Ensure the dataset exists at the specified file_path before execution
- All dependent modules should be properly configured and imported
"""
from pipeline.feature_engineering import FeatureEngineering
from pipeline.ingest import DataIngestor
import pandas as pd
from pipeline.clean import DatasetCleaner
from pipeline.ml.model import PriceModel
from utils.exceptions import PipelineError
from utils.timing import timeit
from utils.logging import setup_logger

from visualization.visualize import DataVisualzation


file_path = 'ebay_merged_data.csv'
connection_string = "postgresql://postgres:0009@localhost:5432/ds_db"
logger = setup_logger()


@timeit
def run_pipeline(source_path: str):
    try:
    # Data ingestor injection and its metrics
        data_ingestor = DataIngestor(source_path)
        raw_df_dict = data_ingestor.fetch_raw_csv_data()
        raw_df = raw_df_dict['dataframe']
        metrics = raw_df_dict['quality_metrics']

        # Data cleaning and its metrics
        dc_results = DatasetCleaner(raw_df).run_pipeline()

        # Visualization
        # vc = DataVisualzation(dc_results)._plot_graph()

        # Feature engineering
        feature_eng = FeatureEngineering(dc_results)
        feature_df = feature_eng.run_features_pipeline()

        # Model training



        return {
            'raw_df': raw_df, 
            'metrics': metrics,
            'cleaned_df': dc_results,
            'feature_df': feature_df
            }
    except PipelineError as e:
        logger.error(f'Error running pipeline with reason{e}')
        raise
    except Exception as e:
        logger.error(f'Fatal error running pipeline with error reason {e}')
        raise

if __name__ == '__main__':
    try:
        pipeline_results = run_pipeline(file_path)
        model = PriceModel(pipeline_results['feature_df']).train()
        
        
    except Exception as e:
        raise RuntimeError(f'Error Running Production Pipeline: {e}')
