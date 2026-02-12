"""
Data Ingestion Pipeline

This script:
1. Loads configuration from params.yaml
2. Fetches raw data from a remote source
3. Performs basic preprocessing
4. Splits data into train and test sets
5. Saves processed data in a DVC-safe structure
"""

import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------

# Logging Configuration

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Log to console (useful during development)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(log_format)

# Log to file (useful for debugging in production)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)

# Avoid duplicate logs
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------------------------------------------------------------------------------------

# Configuration Loader

def load_params(params_path: str) -> dict:
    """
    Load pipeline parameters from a YAML configuration file.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters successfully loaded from %s", params_path)
        return params

    except Exception as e:
        logger.exception("Failed to load parameters")
        raise

# ---------------------------------------------------------------------------------------------

# Data Loader

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file hosted at a given URL.
    """
    try:
        df = pd.read_csv(data_url)
        logger.info("Data successfully loaded from %s", data_url)
        return df

    except Exception as e:
        logger.exception("Failed to load data")
        raise

# ---------------------------------------------------------------------------------------------

# Data Preprocessing

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning and column standardization.
    """
    try:
        # Drop unnecessary unnamed columns
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

        # Rename columns for clarity
        df = df.rename(columns={"v1": "target", "v2": "text"})

        logger.info("Data preprocessing completed successfully")
        return df

    except Exception as e:
        logger.exception("Data preprocessing failed")
        raise

# ---------------------------------------------------------------------------------------------

# Data Persistence (DVC-Safe)


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save processed train and test datasets to disk.
    """
    try:
        raw_data_dir = os.path.join("data", "raw")
        os.makedirs(raw_data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_dir, "test.csv"), index=False)

        logger.info("Train and test data saved to %s", raw_data_dir)

    except Exception as e:
        logger.exception("Failed to save datasets")
        raise

# ---------------------------------------------------------------------------------------------

# Main Pipeline Orchestrator


def main():
    """
    Execute the complete data ingestion pipeline.
    """
    try:
        # Load configuration
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        # Data source
        source_url = (
            "https://raw.githubusercontent.com/"
            "vikashishere/Datasets/main/spam.csv"
        )

        # Pipeline execution
        df = load_data(source_url)
        df = preprocess_data(df)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=1
        )

        save_data(train_df, test_df)

        logger.info("Data ingestion pipeline completed successfully ")

    except Exception as e:
        logger.critical("Data ingestion pipeline failed ")
        raise

# ---------------------------------------------------------------------------------------------

# Script Entry Point

if __name__ == "__main__":
    main()
