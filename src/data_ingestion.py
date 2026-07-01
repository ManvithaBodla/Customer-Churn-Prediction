"""Data ingestion utilities for IBM Telco Customer Churn dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas import DataFrame

DEFAULT_REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

logger = logging.getLogger(__name__)


def configure_logger(level: int = logging.INFO) -> None:
    """Configure a module-level logger for data ingestion."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def load_telco_churn_dataset(source_path: Path) -> DataFrame:
    """Load the Telco Customer Churn dataset from a CSV file.

    Args:
        source_path: Path to the raw dataset CSV.

    Returns:
        A DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the file cannot be parsed as a valid CSV.
    """
    source_path = Path(source_path)
    logger.debug("Loading dataset from %s", source_path)

    if not source_path.exists():
        logger.error("Raw dataset not found at %s", source_path)
        raise FileNotFoundError(f"Raw dataset not found at {source_path}")

    try:
        df = pd.read_csv(source_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Failed to read dataset from %s", source_path)
        raise ValueError(f"Unable to read dataset from {source_path}: {exc}") from exc

    logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))
    return df


def validate_required_columns(df: DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that required columns are present in the dataset.

    Args:
        df: The DataFrame to validate.
        required_columns: Iterable of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")
    logger.info("All required columns are present")


def handle_missing_values(
    df: DataFrame,
    strategy: str = "drop",
    fill_value: Any = "Unknown",
) -> DataFrame:
    """Handle missing values in the dataset.

    Args:
        df: The raw DataFrame.
        strategy: Missing-value strategy. Supported: "drop", "fill".
        fill_value: Value used for filling missing values when strategy is "fill".

    Returns:
        A cleaned DataFrame.

    Raises:
        ValueError: If an unsupported strategy is provided.
    """
    missing_count = int(df.isna().sum().sum())
    logger.debug("Found %d missing values before cleaning", missing_count)

    if missing_count == 0:
        logger.info("No missing values detected")
        return df.copy()

    if strategy == "drop":
        cleaned_df = df.dropna()
        logger.info(
            "Dropped rows with missing values: %d -> %d",
            len(df),
            len(cleaned_df),
        )
    elif strategy == "fill":
        cleaned_df = df.fillna(fill_value)
        logger.info("Filled missing values with %r", fill_value)
    else:
        logger.error("Unsupported missing value strategy: %s", strategy)
        raise ValueError(
            "Unsupported missing value strategy. Use 'drop' or 'fill'."
        )

    if cleaned_df.empty:
        logger.warning("Dataset is empty after missing value handling")

    return cleaned_df


def save_cleaned_dataset(df: DataFrame, destination_path: Path) -> Path:
    """Save the cleaned dataset to a CSV file.

    Args:
        df: The cleaned DataFrame.
        destination_path: Output path for the cleaned CSV.

    Returns:
        The resolved path to the saved CSV.
    """
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(destination_path, index=False)
    except OSError as exc:
        logger.exception("Failed to write cleaned dataset to %s", destination_path)
        raise

    logger.info("Saved cleaned dataset to %s", destination_path)
    return destination_path


def ingest_telco_data(
    raw_path: Path,
    processed_dir: Path,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
    missing_strategy: str = "drop",
    fill_value: Any = "Unknown",
) -> Path:
    """Run the full ingestion flow for the Telco Customer Churn dataset.

    Args:
        raw_path: Path to the raw CSV file.
        processed_dir: Directory to save the cleaned dataset.
        required_columns: Required dataset columns.
        missing_strategy: Strategy for handling missing values.
        fill_value: Value to use when filling missing values.

    Returns:
        The path to the saved cleaned CSV file.
    """
    configure_logger()
    raw_path = Path(raw_path)
    processed_dir = Path(processed_dir)
    logger.info("Starting ingestion for %s", raw_path)

    df = load_telco_churn_dataset(raw_path)
    validate_required_columns(df, required_columns)
    cleaned_df = handle_missing_values(df, strategy=missing_strategy, fill_value=fill_value)

    if cleaned_df.empty:
        logger.error("Cleaned dataset is empty after ingestion")
        raise ValueError("Cleaned dataset is empty after ingestion")

    output_path = processed_dir / f"{raw_path.stem}_cleaned.csv"
    saved_path = save_cleaned_dataset(cleaned_df, output_path)
    logger.info("Ingestion completed successfully")
    return saved_path


if __name__ == "__main__":
    configure_logger()
    default_source = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    default_processed = Path("data/processed")
    try:
        ingest_telco_data(default_source, default_processed)
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise
