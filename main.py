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
from pipeline.ingest import DataIngestor
import pandas as pd
from pipeline.clean import DatasetCleaner
from utils.timing import timeit
pd.set_option('display.max_columns', None)

file_path = 'ebay_merged_data.csv'
connection_string = "postgresql://postgres:0009@localhost:5432/ds_db"


@timeit
def run_pipeline(source_path: str):

    # Data ingestor injection and its metrics
    data_ingestor = DataIngestor(source_path)
    raw_df_dict = data_ingestor.fetch_raw_csv_data()
    raw_df = raw_df_dict['dataframe']
    metrics = raw_df_dict['quality_metrics']

    # Data cleaning and its metrics
    dc_results = DatasetCleaner(raw_df).run_pipeline()

    return {
        'raw_df': raw_df, 
        'metrics': metrics,
        'cleaned_df': dc_results
        }


if __name__ == '__main__':
    try:
        pipeline_results = run_pipeline(file_path)
        print(pipeline_results['cleaned_df'].info())

      
    except Exception as e:
        raise RuntimeError(f'Error Running Production Pipeline: {e}')
